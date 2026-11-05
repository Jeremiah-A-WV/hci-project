"""Drive the mBot2 with finger counts, through the mBlock app.

Hold your hand up with your palm facing the camera and fingers pointing up:

    1 finger -> W (forward)      3 fingers -> D (turn right)
    2 fingers -> S (backward)    4 fingers -> A (turn left)
    fist or no hand -> Space (stop)

This script types those keys into mBlock. In mBlock (Live mode, robot connected),
build these blocks:

    when [w] key pressed     -> moves forward  at 25 RPM for 0.7 secs
    when [s] key pressed     -> moves backward at 25 RPM for 0.7 secs
    when [d] key pressed     -> turns right    at 25 RPM for 0.7 secs
    when [a] key pressed     -> turns left     at 25 RPM for 0.7 secs
    when [space] key pressed -> stop all

Keys in the camera window:   g = arm/pause      q = quit
It starts paused. After arming, show a fist or lower your hand before the robot
will respond, so it can't launch the moment you press g.
"""
import os

# Must be set before MediaPipe loads: run on the CPU and block macOS camera effects.
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

import subprocess
import time

import cv2
import mediapipe as mp

# ---- Settings --------------------------------------------------------------
MBLOCK_APP_NAME = "mBlock"   # the app's name as shown in the macOS menu bar
FOCUS_MBLOCK = True          # bring mBlock to the front before typing each key
TOGGLE_KEY = "g"             # arms/pauses gesture control

# Finger count -> (key typed into mBlock, description)
FINGER_ACTIONS = {
    1: ("w", "forward"),
    2: ("s", "backward"),
    3: ("d", "turn right"),
    4: ("a", "turn left"),
}

STABLE_FRAMES = 8            # frames a pose must hold before it counts
KEY_REPEAT_SEC = 0.5         # re-type the key this often while a pose is held
NO_HAND_STOP_SEC = 1.0       # stop if the hand is gone for this long
LABEL_DISPLAY_SEC = 1.0      # how long the last action stays on screen
# ---------------------------------------------------------------------------

SPACE_KEY_CODE = 49          # macOS key code for the space bar

# (fingertip, middle knuckle) MediaPipe numbers for the index, middle, ring and
# pinky fingers. The thumb is skipped because it's the hardest finger to detect.
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]

reported_errors = set()      # each AppleScript error is only printed once


def run_applescript(script, description):
    """Runs AppleScript and prints a helpful message if it fails."""
    try:
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True, timeout=2)
        error = result.stderr.strip() if result.returncode != 0 else None
    except subprocess.TimeoutExpired:
        error = "timed out"
    if error is None or error in reported_errors:
        return
    reported_errors.add(error)
    print(f"[error] {description} failed: {error}")
    if "not allowed" in error or "1002" in error or "1743" in error:
        print("        -> System Settings > Privacy & Security > Accessibility:"
              " turn ON the app you run this from (VS Code or Terminal).")
    if "Can't get application" in error or "-1728" in error:
        print(f"        -> No app named '{MBLOCK_APP_NAME}'. Change MBLOCK_APP_NAME, or"
              " set FOCUS_MBLOCK = False and click the mBlock window yourself.")


def send_key(key):
    """Types one key into mBlock so its matching block runs."""
    if FOCUS_MBLOCK:
        run_applescript(f'tell application "{MBLOCK_APP_NAME}" to activate',
                        f"bringing {MBLOCK_APP_NAME} to front")
    if key == "space":
        script = f'tell application "System Events" to key code {SPACE_KEY_CODE}'
    else:
        script = f'tell application "System Events" to keystroke "{key}"'
    run_applescript(script, f"typing '{key}'")


def count_fingers(landmarks):
    """Counts raised fingers. A fingertip above its knuckle is up (y grows downward)."""
    return sum(1 for tip, knuckle in FINGERS if landmarks[tip].y < landmarks[knuckle].y)


def send_pose(finger_count):
    """Sends the key for a finger count and returns text describing the action."""
    if finger_count in FINGER_ACTIONS:
        key, description = FINGER_ACTIONS[finger_count]
        send_key(key)
        return f"{finger_count} finger{'s' if finger_count != 1 else ''} -> {description}"
    send_key("space")
    return "Fist -> STOP"


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    control_armed = False
    waiting_for_rest = True     # after arming, wait for a fist or no hand first
    robot_moving = False
    candidate_count = None      # finger count being held (None = no hand)
    candidate_frames = 0        # how many frames it has been held
    active_count = None         # the last finger count acted on
    last_key_time = 0.0
    last_hand_time = time.time()
    action_label, label_hide_time = "", 0.0

    print(__doc__)

    with mp_hands.Hands(max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)   # mirror the picture so it feels natural
            height = frame.shape[0]
            now = time.time()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            finger_count = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                finger_count = count_fingers(hand.landmark)
                last_hand_time = now

            # A pose only counts after it has been steady for a moment.
            if finger_count is not None and finger_count == candidate_count:
                candidate_frames += 1
            else:
                candidate_count = finger_count
                candidate_frames = 1 if finger_count is not None else 0
            pose_is_stable = candidate_count is not None and candidate_frames >= STABLE_FRAMES
            hand_is_resting = candidate_count is None or (pose_is_stable and candidate_count == 0)

            if control_armed and waiting_for_rest and hand_is_resting:
                waiting_for_rest = False
                print("[ready] fist/no hand seen -- finger gestures are live")

            if control_armed and not waiting_for_rest:
                if pose_is_stable and candidate_count != active_count:
                    # A new steady pose: act on it right away.
                    active_count = candidate_count
                    action_label = send_pose(candidate_count)
                    label_hide_time = now + LABEL_DISPLAY_SEC
                    last_key_time = now
                    robot_moving = candidate_count in FINGER_ACTIONS
                    print(action_label)
                elif pose_is_stable and robot_moving and now - last_key_time >= KEY_REPEAT_SEC:
                    # Same pose still held: the mBlock blocks run briefly, so re-send.
                    send_pose(candidate_count)
                    last_key_time = now

                # The hand left the frame while driving, so stop.
                if robot_moving and candidate_count is None and now - last_hand_time > NO_HAND_STOP_SEC:
                    send_key("space")
                    robot_moving = False
                    active_count = 0
                    action_label = "No hand -> STOP"
                    label_hide_time = now + LABEL_DISPLAY_SEC
                    print(action_label)

            # ---- On-screen status (colors are BGR) ----
            if not control_armed:
                status, status_color = f"PAUSED (press '{TOGGLE_KEY}' to arm)", (0, 0, 255)
            elif waiting_for_rest:
                status, status_color = "ARMED - show a fist or lower your hand to start", (0, 255, 255)
            else:
                status, status_color = f"ARMED  (press '{TOGGLE_KEY}' to pause)", (0, 255, 0)
            cv2.putText(frame, f"mBot2: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            seeing = "no hand" if finger_count is None else f"{finger_count} finger(s)"
            cv2.putText(frame, f"Seeing: {seeing}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if now < label_hide_time:
                cv2.putText(frame, action_label, (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "1=fwd 2=back 3=right 4=left  fist=STOP", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("mBot2 Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(TOGGLE_KEY):
                control_armed = not control_armed
                if control_armed:
                    waiting_for_rest = True
                    active_count = None
                elif robot_moving:
                    send_key("space")
                    robot_moving = False
                print(f"[toggle] mBot2 gesture control {'ARMED' if control_armed else 'PAUSED'}")

    # Always try to leave the robot stopped.
    if control_armed:
        send_key("space")
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()