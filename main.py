import os
import time
import subprocess
from collections import deque
import cv2
import mediapipe as mp
import pyautogui

# 1. Forcefully block macOS from injecting background camera portrait/blur layers[cite: 1, 2]
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

# Disable failsafe to prevent the script from crashing if mouse goes to the screen corner
pyautogui.FAILSAFE = False 
pyautogui.PAUSE = 0

# 2. Initialize legacy solutions (Bypasses the C++ tasks backend completely)[cite: 1, 2]
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Landmark indices (MediaPipe Hands topology)
WRIST = 0
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# ---- Gesture tuning knobs -------------------------------------------------
HORIZONTAL_SWIPE_PX = 150       # Distance for left/right slide control[cite: 1]
VERTICAL_SWIPE_PX = 55          # Distance for up/down volume control
GESTURE_COOLDOWN_SEC = 1.0      # Time between triggered gestures[cite: 1]
HISTORY_WINDOW_SEC = 0.5        # Timeframe to evaluate the swipe speed[cite: 2]
VOLUME_STEP = 5                 # % volume change per swipe[cite: 2]
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
            return 50  

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
        return self._apply(self._volume + direction * VOLUME_STEP)

    @property
    def volume(self):
        return self._volume


def is_index_pointing(landmarks):
    """True when the index finger is extended and the other fingers are curled[cite: 2]."""
    index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
    middle_curled = landmarks[MIDDLE_TIP].y > landmarks[MIDDLE_PIP].y
    ring_curled = landmarks[RING_TIP].y > landmarks[RING_PIP].y
    pinky_curled = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
    return index_extended and middle_curled and ring_curled and pinky_curled

def is_whole_hand_open(landmarks):
    """True when the index, middle, ring, and pinky fingers are all extended."""
    index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
    middle_extended = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
    ring_extended = landmarks[RING_TIP].y < landmarks[RING_PIP].y
    pinky_extended = landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y
    return index_extended and middle_extended and ring_extended and pinky_extended


class SwipeDetector:
    """Tracks both X and Y axes to fire presentation or volume controls seamlessly."""
    def __init__(self, on_swipe):
        self.on_swipe = on_swipe
        self.history = deque()  # (timestamp, x_pixel, y_pixel)
        self.last_trigger_time = 0.0

    def reset(self):
        self.history.clear()

    def update(self, tip_x_px, tip_y_px, now, allowed_axis):
        self.history.append((now, tip_x_px, tip_y_px))
        
        # Prune old tracking data
        while self.history and now - self.history[0][0] > HISTORY_WINDOW_SEC:
            self.history.popleft()

        if now - self.last_trigger_time < GESTURE_COOLDOWN_SEC:
            return None
        if len(self.history) < 2:
            return None

        oldest_x = self.history[0][1]
        oldest_y = self.history[0][2]
        
        dx = tip_x_px - oldest_x
        dy = tip_y_px - oldest_y  

        # Evaluate against the explicitly allowed axis
        if allowed_axis == "horizontal" and abs(dx) > abs(dy) and abs(dx) >= HORIZONTAL_SWIPE_PX:
            direction = "right" if dx > 0 else "left"
            self.last_trigger_time = now
            self.history.clear()
            self.on_swipe(direction)
            return direction
            
        elif allowed_axis == "vertical" and abs(dy) >= abs(dx) and abs(dy) >= VERTICAL_SWIPE_PX:
            direction = "up" if dy < 0 else "down"
            self.last_trigger_time = now
            self.history.clear()
            self.on_swipe(direction)
            return direction
                
        return None

# 3. Target your Mac FaceTime camera using AVFOUNDATION backend[cite: 1, 2]
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
volume_controller = MacVolumeController()

last_swipe_label = ""
last_swipe_label_until = 0.0

def handle_swipe(direction):
    global last_swipe_label, last_swipe_label_until
    
    if direction == "up":
        new_vol = volume_controller.step(+1)
        last_swipe_label = f"Volume UP -> {new_vol}%"
        print(last_swipe_label)
    elif direction == "down":
        new_vol = volume_controller.step(-1)
        last_swipe_label = f"Volume DOWN -> {new_vol}%"
        print(last_swipe_label)
    elif direction == "right":
        pyautogui.press('right')
        last_swipe_label = "Swiped Right! -> Next Slide"
        print(last_swipe_label)
    elif direction == "left":
        pyautogui.press('left')
        last_swipe_label = "Swiped Left! -> Prev Slide"
        print(last_swipe_label)
        
    last_swipe_label_until = time.time() + 1.0


swipe_detector = SwipeDetector(on_swipe=handle_swipe)

# Configure the tracking instance[cite: 1, 2]
with mp_hands.Hands(
    static_image_mode=False,        
    max_num_hands=2,
    model_complexity=1,             
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip horizontally for natural mirror view[cite: 1, 2]
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Convert OpenCV BGR frame to MediaPipe required RGB format[cite: 1, 2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        tracking_this_frame = False

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                hand_label = "Hand"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    hand_label = results.multi_handedness[idx].classification[0].label

                # 4. Use MediaPipe's drawing utility[cite: 1, 2]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                wrist_landmark = hand_landmarks.landmark[WRIST]
                cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
                cv2.putText(frame, hand_label, (cx - 20, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Evaluate hand states
                is_pointing = is_index_pointing(hand_landmarks.landmark)
                is_open = is_whole_hand_open(hand_landmarks.landmark)

                # Track gestures based on the specific hand shape
                if is_pointing or is_open:
                    tracking_this_frame = True
                    tip = hand_landmarks.landmark[INDEX_TIP]
                    tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                    
                    # Visual feedback: Blue = Volume (Pointing), Yellow = Slides (Open Hand)
                    color = (255, 0, 0) if is_pointing else (0, 255, 255) 
                    cv2.circle(frame, (tip_x, tip_y), 15, color, cv2.FILLED)
                    
                    allowed_axis = "vertical" if is_pointing else "horizontal"
                    swipe_detector.update(tip_x, tip_y, now, allowed_axis)

        if not tracking_this_frame:
            swipe_detector.reset()

        # HUD feedback[cite: 2]
        cv2.putText(frame, f"Volume: {volume_controller.volume}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if last_swipe_label and now < last_swipe_label_until:
            cv2.putText(frame, last_swipe_label, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Unified Mac Gesture Controller', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()