import os
import time
import subprocess
from collections import deque

# 1. Forcefully block macOS from injecting background camera portrait/blur layers
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

import cv2
import mediapipe as mp

# 2. Initialize legacy solutions (Bypasses the C++ tasks backend completely)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Standard 21-point hand skeleton connections map automatically in legacy
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Landmark indices (MediaPipe Hands topology)
WRIST = 0
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# ---- Gesture tuning knobs -------------------------------------------------
SWIPE_PIXEL_THRESHOLD = 55      # how far the fingertip must travel (px) to count as a swipe
GESTURE_COOLDOWN_SEC = 0.45     # minimum time between two triggered volume changes
HISTORY_WINDOW_SEC = 0.5        # how far back in time we look for the swipe distance
VOLUME_STEP = 5                 # % volume change per swipe
# ---------------------------------------------------------------------------


class MacVolumeController:
    """Wraps `osascript` calls so we can read/set the macOS output volume."""

    def __init__(self):
        self._volume = self._get_system_volume()

    @staticmethod
    def _get_system_volume():
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=2,
            )
            return int(result.stdout.strip())
        except Exception:
            return 50  # sane fallback if AppleScript call fails

    def _apply(self, new_volume):
        new_volume = max(0, min(100, new_volume))
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {new_volume}"],
                timeout=2,
            )
            self._volume = new_volume
        except Exception as exc:
            print(f"[volume] failed to set volume: {exc}")
        return self._volume

    def step(self, direction):
        """direction: +1 raises volume, -1 lowers it."""
        return self._apply(self._volume + direction * VOLUME_STEP)

    @property
    def volume(self):
        return self._volume


def is_index_pointing(landmarks):
    """True when the index finger is extended and the other fingers are curled
    (a 'pointing' pose), so open-palm or fist gestures don't trigger a swipe."""
    index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
    middle_curled = landmarks[MIDDLE_TIP].y > landmarks[MIDDLE_PIP].y
    ring_curled = landmarks[RING_TIP].y > landmarks[RING_PIP].y
    pinky_curled = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
    return index_extended and middle_curled and ring_curled and pinky_curled


class SwipeDetector:
    """Tracks the index fingertip's y position while it is 'pointing' and
    fires a callback once a fast enough vertical swipe is detected."""

    def __init__(self, on_swipe):
        self.on_swipe = on_swipe
        self.history = deque()  # (timestamp, y_pixel)
        self.last_trigger_time = 0.0

    def reset(self):
        self.history.clear()

    def update(self, fingertip_y_px, now):
        self.history.append((now, fingertip_y_px))
        while self.history and now - self.history[0][0] > HISTORY_WINDOW_SEC:
            self.history.popleft()

        if now - self.last_trigger_time < GESTURE_COOLDOWN_SEC:
            return None
        if len(self.history) < 2:
            return None

        oldest_y = self.history[0][1]
        dy = fingertip_y_px - oldest_y  # image y grows downward

        if abs(dy) < SWIPE_PIXEL_THRESHOLD:
            return None

        direction = "up" if dy < 0 else "down"
        self.last_trigger_time = now
        self.history.clear()
        self.on_swipe(direction)
        return direction


# 3. Target your Mac FaceTime camera using AVFOUNDATION backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

volume_controller = MacVolumeController()

last_swipe_label = ""
last_swipe_label_until = 0.0


def handle_swipe(direction):
    global last_swipe_label, last_swipe_label_until
    if direction == "up":
        new_vol = volume_controller.step(+1)
        last_swipe_label = f"Volume UP -> {new_vol}%"
        gesture_code = 2
    else:
        new_vol = volume_controller.step(-1)
        last_swipe_label = f"Volume DOWN -> {new_vol}%"
        gesture_code = 3
    last_swipe_label_until = time.time() + 1.0
    print(f"[ACTION COMPLETE] {last_swipe_label} | gesture_code={gesture_code}")
    print(gesture_code)


swipe_detector = SwipeDetector(on_swipe=handle_swipe)

# Configure the tracking instance
with mp_hands.Hands(
    static_image_mode=False,        # Optimized for continuous video tracking
    max_num_hands=2,
    model_complexity=1,             # 1 is standard, lightweight, and stable
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Convert OpenCV BGR frame to MediaPipe required RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance optimization: Mark image as flag non-writeable to speed up pass-by-reference
        rgb_frame.flags.writeable = False

        # Process the frame (Legacy infers timestamp tracking out of the box)
        results = hands.process(rgb_frame)

        # Draw overlays if hands are tracked
        rgb_frame.flags.writeable = True

        pointing_this_frame = False

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # Fetch handedness (Left vs Right)
                hand_label = "Hand"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    hand_label = results.multi_handedness[idx].classification[0].label

                # 4. Use MediaPipe's robust default drawing utility (Prevents shape errors)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get pixel coordinates of the wrist (landmark 0) to append text label
                wrist_landmark = hand_landmarks.landmark[WRIST]
                cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
                cv2.putText(frame, hand_label, (cx - 20, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- Volume swipe gesture: extended index finger, others curled ---
                if is_index_pointing(hand_landmarks.landmark):
                    pointing_this_frame = True
                    tip = hand_landmarks.landmark[INDEX_TIP]
                    tip_px = (int(tip.x * w), int(tip.y * h))
                    cv2.circle(frame, tip_px, 10, (0, 255, 0), -1)
                    swipe_detector.update(tip_px[1], now)

        if not pointing_this_frame:
            swipe_detector.reset()

        # HUD: current volume + last swipe event
        cv2.putText(frame, f"Volume: {volume_controller.volume}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if last_swipe_label and now < last_swipe_label_until:
            cv2.putText(frame, last_swipe_label, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display window
        cv2.imshow('MediaPipe Legacy Hand Tracker', frame)

        # Press 'q' to exit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()