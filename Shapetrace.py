"""Pick a shape for the mBot2 by tracing it in the air or holding up fingers.

  Trace: point with one index finger and draw the shape, finishing near where
         you started. Lower or open your hand to finish. Recognizes a circle
         and polygons with 3 to 8 sides (use fingers if an octagon reads as a circle).

  Fingers: hold up 3 to 8 fingers across one or two hands, thumbs included, until
         the bar fills.
         3 = triangle  4 = rectangle  5 = pentagon  6 = hexagon  7 = heptagon  8 = octagon

The chosen shape is drawn in green and the mBot2 drives it right away, so put the
robot on the floor with clear space around it. New picks are ignored until it finishes.

Keys in the camera window:   s = stop robot      c = clear      q = quit

Before running, turn the robot on and disconnect it in mBlock, because only one
program can hold its Bluetooth link.
"""
import os
import sys

# Must be set before MediaPipe loads: run on the CPU and block macOS camera effects.
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

# If the play button started a Python without these packages, restart this script with
# the project's virtual environment.
VENV_PYTHON = "/Users/gomezfile/Downloads/HCI_Work/hci-project/.venv/bin/python"
try:
    import cv2
    from mbot2 import MBot2
except ModuleNotFoundError:
    if os.path.exists(VENV_PYTHON) and not os.environ.get("HCI_RELAUNCHED"):
        os.environ["HCI_RELAUNCHED"] = "1"
        os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
    raise

import math
import threading
import time
from collections import deque

import numpy as np

# ---- Settings --------------------------------------------------------------
CIRCLE = 0   # circles are stored as "0 sides"
SHAPE_NAMES = {CIRCLE: "circle", 3: "triangle", 4: "rectangle", 5: "pentagon",
               6: "hexagon", 7: "heptagon", 8: "octagon"}

FINGER_HOLD_SEC = 1.5      # how long to hold a finger count to pick it
TRACE_END_SEC = 0.35       # stop pointing for this long and the trace is finished
PICK_COOLDOWN_SEC = 2.0    # pause after a pick so it can't instantly repeat
MESSAGE_DISPLAY_SEC = 2.5  # how long an error message stays on screen
TIP_SMOOTHING = 0.5        # 0 to 1; higher = smoother but laggier fingertip trail
POLYGON_SIDE_CM = 20       # side length the robot drives for polygons
CIRCLE_DIAMETER_CM = 40    # diameter the robot drives for a circle

# Robot driving
ROBOT_NAME = "Makeblock_LE08D1F96D2376"   # connect only to this robot, not a classmate's
DRIVE_SPEED = 30           # wheel speed in RPM while drawing shapes
RIGHT_WHEEL_SIGN = -1      # the right motor is mirrored; same value as mbot2_wasd.py
CM_PER_SECOND = 10.0       # CALIBRATE: distance driven per second at DRIVE_SPEED
WHEEL_GAP_CM = 12.0        # CALIBRATE: distance between the wheels; adjusts circle size
MOVE_PAUSE_SEC = 0.3       # short rest between moves so the robot settles

# Trace recognizer tuning
MIN_TRACE_POINTS = 15        # fewer fingertip points than this is too short
MIN_TRACE_SIZE_PIXELS = 60   # a smaller trace (corner to corner) is too small
MAX_CLOSING_GAP = 0.35       # allowed start-to-end gap, as a fraction of trace size
RESAMPLED_POINTS = 120       # the trace is redrawn with this many evenly spaced points
SMOOTHING_WINDOW = 5         # points averaged together to remove hand jitter
TURN_LOOKAROUND = 6          # points before and after used to measure each turn
CIRCLE_MAX_UNEVENNESS = 0.55 # turning more even than this means it's a circle
CORNER_TOLERANCE = 0.022     # how closely the polygon follows the trace
MERGE_ANGLE_DEGREES = 160.0  # corners flatter than this are removed
MERGE_DISTANCE = 0.07        # corners closer than this fraction of the perimeter merge
# ---------------------------------------------------------------------------

# (fingertip, middle knuckle) MediaPipe numbers for the index, middle, ring, pinky.
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))
WRIST, THUMB_KNUCKLE, THUMB_TIP, INDEX_TIP, INDEX_BASE, MIDDLE_BASE, PINKY_BASE = 0, 3, 4, 8, 5, 9, 17
FINGER_EXTENDED_RATIO = 1.1  # tip must be this much farther from the wrist than the knuckle


# ================================ Robot plan =================================

def robot_plan(sides):
    """Returns the list of moves the mBot2 makes to drive this shape."""
    if sides == CIRCLE:
        return [("circle", CIRCLE_DIAMETER_CM)]
    plan = []
    for _ in range(sides):
        plan.append(("forward_cm", POLYGON_SIDE_CM))
        plan.append(("turn_left_deg", round(360 / sides, 1)))
    return plan


class ShapeDriver:
    """Connects to the mBot2 and drives shapes on a background thread, so the
    camera keeps running while the robot moves."""

    def __init__(self):
        print("Connecting over Bluetooth...")
        self.robot = MBot2(connect=False)
        # Only accept this robot. macOS can report its full name in either of two
        # places, so check both.
        self.robot._bot._match = lambda device, advertisement: ROBOT_NAME in (
            device.name, advertisement.local_name)
        self.robot.connect()
        self.stop_requested = threading.Event()
        self.drive_thread = None

    def is_driving(self):
        return self.drive_thread is not None and self.drive_thread.is_alive()

    def start_shape(self, sides):
        """Starts driving a shape. Returns False if the robot is already busy."""
        if self.is_driving():
            return False
        self.stop_requested.clear()
        self.drive_thread = threading.Thread(target=self.drive_plan, args=(robot_plan(sides),),
                                             daemon=True)
        self.drive_thread.start()
        return True

    def stop(self):
        """Stops the robot immediately and cancels the rest of the shape."""
        self.stop_requested.set()
        self.robot.stop()

    def close(self):
        """Stops the robot and disconnects (disconnecting also stops the motors)."""
        self.stop_requested.set()
        if self.is_driving():
            self.drive_thread.join(timeout=3)
        self.robot.disconnect()

    def drive_plan(self, plan):
        """Runs on the background thread: performs each move in order."""
        try:
            for move, amount in plan:
                if move == "forward_cm":
                    self.drive_wheels(DRIVE_SPEED, DRIVE_SPEED, amount / CM_PER_SECOND)
                elif move == "turn_left_deg":
                    self.robot.turn(-amount)   # the library treats negative as left
                else:
                    self.drive_circle(amount)
                # wait() returns True as soon as a stop is requested.
                if self.stop_requested.wait(MOVE_PAUSE_SEC):
                    return
        except Exception as error:   # e.g. the Bluetooth link dropped mid-shape
            print(f"[robot] driving failed: {error}")

    def drive_wheels(self, left_speed, right_speed, seconds):
        """Sets both wheels, waits, then stops. A stop request ends the wait early."""
        self.robot.drive(left_speed, RIGHT_WHEEL_SIGN * right_speed)
        self.stop_requested.wait(seconds)
        self.robot.stop()

    def drive_circle(self, diameter_cm):
        """Curves left all the way round: the outer (right) wheel travels farther
        than the inner (left) wheel, in proportion to their distances from the center."""
        radius = diameter_cm / 2
        left_speed = round(DRIVE_SPEED * (radius - WHEEL_GAP_CM / 2) / radius)
        right_speed = round(DRIVE_SPEED * (radius + WHEEL_GAP_CM / 2) / radius)
        self.drive_wheels(left_speed, right_speed, math.pi * diameter_cm / CM_PER_SECOND)


def on_shape_selected(sides, source, driver):
    """Runs once per picked shape: prints the plan and starts the robot driving it."""
    print(f"\n[shape] {SHAPE_NAMES[sides].upper()} (from {source})")
    if not driver.start_shape(sides):
        print("    ignored: the robot is still driving (press s to stop it)")
        return
    for move, amount in robot_plan(sides):
        print(f"    {move:<14} {amount}")


# ============================ Trace recognizer ===============================

def trim_overshoot(points):
    """Cuts the trace where it comes back closest to its start."""
    search_start = int(len(points) * 0.6)
    distances_to_start = np.linalg.norm(points[search_start:] - points[0], axis=1)
    return points[: search_start + int(np.argmin(distances_to_start)) + 1]


def resample_loop(points, point_count):
    """Redraws a closed trace with evenly spaced points, so hand speed doesn't matter."""
    points = np.vstack([points, points[:1]])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate([[True], segment_lengths > 1e-6])]   # drop duplicates
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance_along = np.concatenate([[0], np.cumsum(segment_lengths)])
    targets = np.linspace(0, distance_along[-1], point_count, endpoint=False)
    return np.column_stack([np.interp(targets, distance_along, points[:, 0]),
                            np.interp(targets, distance_along, points[:, 1])])


def smooth_loop(points, window):
    """Averages each point with its neighbors around the closed loop."""
    half_window = window // 2
    return sum(np.roll(points, offset, axis=0)
               for offset in range(-half_window, half_window + 1)) / (2 * half_window + 1)


def turning_angles(points, lookaround):
    """Returns how sharply the path turns at each point, in degrees."""
    incoming = points - np.roll(points, lookaround, axis=0)
    outgoing = np.roll(points, -lookaround, axis=0) - points
    change = np.degrees(np.arctan2(outgoing[:, 1], outgoing[:, 0])
                        - np.arctan2(incoming[:, 1], incoming[:, 0]))
    return np.abs((change + 180) % 360 - 180)


def corner_angle(previous_corner, corner, next_corner):
    """Returns the inside angle at a corner in degrees (180 = perfectly straight)."""
    side_a, side_b = previous_corner - corner, next_corner - corner
    cosine = np.dot(side_a, side_b) / (np.linalg.norm(side_a) * np.linalg.norm(side_b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def clean_corners(corners, perimeter):
    """Merges corners that are too close together and removes nearly straight ones."""
    corners = [np.asarray(corner, float) for corner in corners]
    changed = True
    while changed and len(corners) > 3:
        changed = False
        count = len(corners)
        for i in range(count):
            next_i = (i + 1) % count
            if np.linalg.norm(corners[i] - corners[next_i]) < MERGE_DISTANCE * perimeter:
                corners[i] = (corners[i] + corners[next_i]) / 2
                del corners[next_i]
                changed = True
                break
        if changed:
            continue
        angles = [corner_angle(corners[i - 1], corners[i], corners[(i + 1) % count])
                  for i in range(count)]
        flattest = int(np.argmax(angles))
        if angles[flattest] > MERGE_ANGLE_DEGREES:
            del corners[flattest]
            changed = True
    return corners


def classify_trace(trace):
    """Returns (sides, message, outline). sides is CIRCLE, 3 to 8, or None if unrecognized."""
    points = np.asarray(trace, dtype=float)
    if len(points) < MIN_TRACE_POINTS:
        return None, "trace too short", None
    trace_size = float(np.hypot(*np.ptp(points, axis=0)))
    if trace_size < MIN_TRACE_SIZE_PIXELS:
        return None, "trace too small - draw bigger", None
    if np.linalg.norm(points[0] - points[-1]) > MAX_CLOSING_GAP * trace_size:
        return None, "not closed - finish near where you started", None

    loop = smooth_loop(resample_loop(trim_overshoot(points), RESAMPLED_POINTS), SMOOTHING_WINDOW)

    # Circles turn evenly all the way round; polygons turn only at their corners.
    turns = turning_angles(loop, TURN_LOOKAROUND)
    unevenness = float(np.std(turns) / (np.mean(turns) + 1e-9))

    hull = cv2.convexHull(loop.astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    rough_corners = cv2.approxPolyDP(hull, CORNER_TOLERANCE * perimeter, True).reshape(-1, 2)
    corners = clean_corners(rough_corners, perimeter)
    sides = len(corners)

    if unevenness < CIRCLE_MAX_UNEVENNESS or sides > 8:
        return CIRCLE, SHAPE_NAMES[CIRCLE], loop
    if sides < 3:
        return None, "couldn't tell - try sharper corners", loop
    return sides, SHAPE_NAMES[sides], np.array(corners)


# ============================= Reading the hand ==============================

def distance(point_a, point_b):
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def fingers_extended(landmarks):
    """Returns True/False for the index, middle, ring and pinky. Comparing distances
    from the wrist, rather than up/down, works at any hand angle."""
    wrist = landmarks[WRIST]
    return [distance(landmarks[tip], wrist) > distance(landmarks[knuckle], wrist) * FINGER_EXTENDED_RATIO
            for tip, knuckle in FINGERS]


def count_fingers(landmarks):
    """Counts the extended fingers on one hand, including the thumb."""
    palm_length = distance(landmarks[WRIST], landmarks[MIDDLE_BASE])
    thumb_out = (distance(landmarks[THUMB_TIP], landmarks[PINKY_BASE])
                 > distance(landmarks[THUMB_KNUCKLE], landmarks[PINKY_BASE])
                 and distance(landmarks[THUMB_TIP], landmarks[INDEX_BASE]) > 0.5 * palm_length)
    return sum(fingers_extended(landmarks)) + (1 if thumb_out else 0)


def is_pointing(landmarks):
    """Index finger out, the other three curled (the thumb is ignored)."""
    return fingers_extended(landmarks) == [True, False, False, False]


# ================================= Drawing ===================================

def regular_polygon(sides, center, radius):
    """Returns the corners of an even polygon with one corner pointing up."""
    first_angle = -math.pi / 2
    return np.array([(center[0] + radius * math.cos(first_angle + 2 * math.pi * i / sides),
                      center[1] + radius * math.sin(first_angle + 2 * math.pi * i / sides))
                     for i in range(sides)])


def draw_shape(frame, sides, outline):
    """Draws the picked shape. outline is None for a finger pick: a clean shape goes mid-screen."""
    height, width = frame.shape[:2]
    if outline is None:
        center, radius = (width // 2, height // 2), min(width, height) // 4
        corners = regular_polygon(sides, center, radius).astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 4)
    elif sides == CIRCLE:
        center = outline.mean(axis=0)
        radius = float(np.linalg.norm(outline - center, axis=1).mean())
        cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 4)
    else:
        cv2.polylines(frame, [outline.astype(np.int32)], True, (0, 255, 0), 4)
        for corner in outline.astype(int):
            cv2.circle(frame, (int(corner[0]), int(corner[1])), 7, (0, 200, 255), cv2.FILLED)


# ================================== Main =====================================

def run_camera(driver):
    """Runs the camera loop until q is pressed, sending picked shapes to the driver."""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    trace = []                     # fingertip points of the current trace
    smoothed_tip = None            # fingertip position after smoothing
    last_pointing_time = 0.0
    held_count, hold_start_time = None, 0.0
    recent_counts = deque(maxlen=9)
    last_picked_count = None       # the count must change before it can be picked again
    cooldown_end_time = 0.0
    picked_shape = None            # (sides, outline, label)
    message, message_hide_time = "", 0.0

    print(__doc__)

    with mp_hands.Hands(max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)   # mirror the picture so it feels natural
            height, width = frame.shape[:2]
            now = time.time()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detected_hands = results.multi_hand_landmarks or []
            for hand in detected_hands:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())

            cooling_down = now < cooldown_end_time
            tracing_now = (len(detected_hands) == 1 and not cooling_down
                           and is_pointing(detected_hands[0].landmark))

            # ---- Trace mode ----
            if tracing_now:
                tip = detected_hands[0].landmark[INDEX_TIP]
                tip_position = np.array([tip.x * width, tip.y * height])
                smoothed_tip = tip_position if smoothed_tip is None else (
                    TIP_SMOOTHING * smoothed_tip + (1 - TIP_SMOOTHING) * tip_position)
                if not trace:
                    picked_shape = None   # starting a new trace clears the old shape
                trace.append(smoothed_tip.copy())
                last_pointing_time = now
                held_count = None
            elif trace and now - last_pointing_time > TRACE_END_SEC:
                sides, trace_message, outline = classify_trace(trace)
                if sides is None:
                    message, message_hide_time = trace_message, now + MESSAGE_DISPLAY_SEC
                    print(f"[trace] not recognized: {trace_message}")
                else:
                    picked_shape = (sides, outline, f"Traced: {SHAPE_NAMES[sides]}")
                    on_shape_selected(sides, "trace", driver)
                    cooldown_end_time = now + PICK_COOLDOWN_SEC
                trace, smoothed_tip = [], None

            # ---- Finger mode ----
            if not tracing_now and not trace:
                recent_counts.append(sum(count_fingers(hand.landmark) for hand in detected_hands)
                                     if detected_hands else None)
                # The most common recent count, so one flickery frame can't reset the timer.
                steady_count = max(set(recent_counts), key=list(recent_counts).count)
                if steady_count != held_count:
                    held_count, hold_start_time = steady_count, now
                if steady_count != last_picked_count:
                    last_picked_count = None
                seconds_held = now - hold_start_time
                if (steady_count in SHAPE_NAMES and steady_count != CIRCLE and not cooling_down
                        and steady_count != last_picked_count):
                    progress = min(1.0, seconds_held / FINGER_HOLD_SEC)
                    cv2.rectangle(frame, (10, height - 70), (10 + int(300 * progress), height - 55), (0, 255, 0), -1)
                    cv2.rectangle(frame, (10, height - 70), (310, height - 55), (255, 255, 255), 1)
                    cv2.putText(frame, f"{steady_count} fingers -> {SHAPE_NAMES[steady_count]}",
                                (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if seconds_held >= FINGER_HOLD_SEC:
                        picked_shape = (steady_count, None, f"Fingers: {SHAPE_NAMES[steady_count]}")
                        on_shape_selected(steady_count, f"{steady_count} fingers", driver)
                        last_picked_count = steady_count
                        cooldown_end_time = now + PICK_COOLDOWN_SEC

            # ---- On-screen status (colors are BGR) ----
            if len(trace) > 1:
                cv2.polylines(frame, [np.array(trace, dtype=np.int32)], False, (255, 120, 0), 3)
            if picked_shape:
                draw_shape(frame, picked_shape[0], picked_shape[1])
                cv2.putText(frame, picked_shape[2], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if now < message_hide_time:
                cv2.putText(frame, message, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if tracing_now:
                status = "TRACING... open/lower your hand to finish"
            elif not detected_hands:
                status = "Point to trace, or hold up 3-8 fingers"
            else:
                total_fingers = sum(count_fingers(hand.landmark) for hand in detected_hands)
                status = f"Seeing {total_fingers} finger(s) on {len(detected_hands)} hand(s)"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if driver.is_driving():
                cv2.putText(frame, "ROBOT DRIVING - press s to stop", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, "s = stop robot   c = clear   q = quit", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Shape Tracer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                driver.stop()
                print("[robot] stopped")
            if key == ord("c"):
                picked_shape, trace, smoothed_tip, message = None, [], None, ""

    camera.release()
    cv2.destroyAllWindows()


def main():
    driver = ShapeDriver()   # connect first, so a failed connection stops here
    try:
        run_camera(driver)
    finally:
        print("Stopping and disconnecting...")
        driver.close()


if __name__ == "__main__":
    main()