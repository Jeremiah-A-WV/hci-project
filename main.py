"""Hand-gesture control for the Mac, using the webcam.

    Point with index finger, move up/down   -> system volume up/down
    Open hand, move up/down                 -> next/previous TikTok video
    Open hand, move left/right              -> next/previous YouTube video or slide

Keys in the camera window:   g = pause/resume gestures      q = quit
"""
import os

# Must be set before MediaPipe loads: run on the CPU and block macOS camera effects.
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

import subprocess
import time
from collections import deque

import cv2
import mediapipe as mp

# ---- Settings --------------------------------------------------------------
HORIZONTAL_SWIPE_PIXELS = 150   # how far a hand must travel for a left/right swipe
VERTICAL_SWIPE_PIXELS = 55      # how far a hand must travel for an up/down swipe
SWIPE_WINDOW_SEC = 0.5          # the travel must happen within this much time
SWIPE_COOLDOWN_SEC = 1.0        # minimum wait between two swipes
VOLUME_STEP_PERCENT = 5         # volume change per swipe
LABEL_DISPLAY_SEC = 1.0         # how long the last action stays on screen

HORIZONTAL_SWIPE_TARGET = "youtube"   # "youtube" or "slides"
TIKTOK_NEXT_BUTTON = "[aria-label='Go to next video']"       # CSS selectors for
TIKTOK_PREVIOUS_BUTTON = "[aria-label='Go to previous video']"   # TikTok's buttons
TOGGLE_KEY = "g"                # pauses/resumes all gesture actions
# ---------------------------------------------------------------------------

# macOS key codes for the arrow keys.
ARROW_KEY_CODES = {"right": 124, "left": 123, "up": 126, "down": 125}

# MediaPipe's numbers for the fingertip and middle knuckle (PIP) of each finger.
INDEX_FINGER = (8, 6)
MIDDLE_FINGER = (12, 10)
RING_FINGER = (16, 14)
PINKY_FINGER = (20, 18)
INDEX_TIP = INDEX_FINGER[0]


# ============================ Controlling the Mac ============================

def run_applescript(script):
    """Runs AppleScript. Returns its output, or None if it failed."""
    try:
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True, timeout=2)
    except subprocess.TimeoutExpired:
        print("[applescript] timed out")
        return None
    if result.returncode != 0:
        print(f"[applescript] failed: {result.stderr.strip()}")
        return None
    return result.stdout.strip()


def focus_chrome():
    """Brings Chrome to the front, because keystrokes go to the frontmost app."""
    run_applescript('tell application "Google Chrome" to activate')


def press_arrow_key(direction):
    run_applescript(f'tell application "System Events" to key code {ARROW_KEY_CODES[direction]}')


def press_shift_and_key(letter):
    """Sends Shift+letter (YouTube uses Shift+N and Shift+P)."""
    run_applescript(
        f'tell application "System Events" to keystroke "{letter}" using {{shift down}}')


def click_chrome_button(css_selector):
    """Clicks a button on the current Chrome page. Returns True if it was clicked.

    Requires Chrome's View > Developer > Allow JavaScript from Apple Events.
    """
    focus_chrome()
    javascript = (
        '(function(){'
        f'var el = document.querySelector("{css_selector}");'
        'if (!el) return "NOT_FOUND";'
        'el.click();'
        'return "CLICKED";'
        '})();'
    )
    # Escape the JavaScript's quotes so they don't end the AppleScript string.
    escaped_javascript = javascript.replace("\\", "\\\\").replace('"', '\\"')
    outcome = run_applescript(f'''
    tell application "Google Chrome"
        tell active tab of front window
            execute javascript "{escaped_javascript}"
        end tell
    end tell
    ''')
    if outcome == "CLICKED":
        return True
    print(f"[scroll] could not click {css_selector!r}. Check that TikTok is the front tab "
          'and "Allow JavaScript from Apple Events" is on.')
    return False


class VolumeController:
    """Remembers the macOS output volume and changes it in steps."""

    def __init__(self):
        current_volume = run_applescript("output volume of (get volume settings)")
        try:
            self.volume = int(current_volume)
        except (TypeError, ValueError):   # no reading available, e.g. HDMI output
            self.volume = 50

    def step(self, direction):
        """Moves the volume up (direction = +1) or down (direction = -1)."""
        new_volume = max(0, min(100, self.volume + direction * VOLUME_STEP_PERCENT))
        if run_applescript(f"set volume output volume {new_volume}") is not None:
            self.volume = new_volume
        return self.volume


# ============================ Reading the hand ===============================

def finger_is_up(landmarks, finger):
    """A finger is up when its tip is above its knuckle (y grows downward)."""
    tip, knuckle = finger
    return landmarks[tip].y < landmarks[knuckle].y


def is_pointing(landmarks):
    """Index finger up, the other three fingers curled."""
    return (finger_is_up(landmarks, INDEX_FINGER)
            and not finger_is_up(landmarks, MIDDLE_FINGER)
            and not finger_is_up(landmarks, RING_FINGER)
            and not finger_is_up(landmarks, PINKY_FINGER))


def is_open_hand(landmarks):
    """All four fingers up (the thumb is ignored)."""
    return all(finger_is_up(landmarks, finger)
               for finger in (INDEX_FINGER, MIDDLE_FINGER, RING_FINGER, PINKY_FINGER))


class SwipeDetector:
    """Watches recent fingertip positions and reports when they form a swipe."""

    def __init__(self):
        self.recent_positions = deque()   # (time, x, y) of the fingertip
        self.last_swipe_time = 0.0

    def reset(self):
        self.recent_positions.clear()

    def update(self, x, y, now, vertical_only):
        """Adds a position. Returns "up", "down", "left", "right", or None."""
        self.recent_positions.append((now, x, y))
        while now - self.recent_positions[0][0] > SWIPE_WINDOW_SEC:
            self.recent_positions.popleft()

        if now - self.last_swipe_time < SWIPE_COOLDOWN_SEC or len(self.recent_positions) < 2:
            return None

        # Compare where the fingertip is now with where it was at the window's start.
        _, start_x, start_y = self.recent_positions[0]
        moved_x, moved_y = x - start_x, y - start_y

        direction = None
        if (not vertical_only and abs(moved_x) > abs(moved_y)
                and abs(moved_x) >= HORIZONTAL_SWIPE_PIXELS):
            direction = "right" if moved_x > 0 else "left"
        elif abs(moved_y) >= abs(moved_x) and abs(moved_y) >= VERTICAL_SWIPE_PIXELS:
            direction = "up" if moved_y < 0 else "down"

        if direction:
            self.last_swipe_time = now
            self.recent_positions.clear()
        return direction


# ============================ Acting on a swipe ==============================

def adjust_volume(direction, volume_controller):
    if direction == "up":
        return f"Volume UP -> {volume_controller.step(+1)}%"
    return f"Volume DOWN -> {volume_controller.step(-1)}%"


def scroll_tiktok(direction):
    """Moving the hand up acts like flicking a phone: it goes to the next video."""
    go_next = direction == "up"
    button = TIKTOK_NEXT_BUTTON if go_next else TIKTOK_PREVIOUS_BUTTON
    if not click_chrome_button(button):
        focus_chrome()                          # arrow keys are the backup plan
        press_arrow_key("down" if go_next else "up")
    return f"Scrolled -> {'Next' if go_next else 'Previous'} (TikTok)"


def change_video_or_slide(direction):
    if HORIZONTAL_SWIPE_TARGET == "youtube":
        focus_chrome()
        if direction == "right":
            press_shift_and_key("n")
            return "YouTube -> Next video"
        press_shift_and_key("p")
        return "YouTube -> Previous video (playlist only)"
    press_arrow_key(direction)
    return f"Swiped {direction.capitalize()}! -> {'Next' if direction == 'right' else 'Prev'} Slide"


# ================================== Main =====================================

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    volume_controller = VolumeController()
    swipe_detector = SwipeDetector()
    gestures_enabled = True
    action_label, label_hide_time = "", 0.0

    with mp_hands.Hands(max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)   # mirror the picture so it feels natural
            height, width = frame.shape[:2]
            now = time.time()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hand_is_gesturing = False
            for hand in results.multi_hand_landmarks or []:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                landmarks = hand.landmark
                pointing = is_pointing(landmarks)
                if not pointing and not is_open_hand(landmarks):
                    continue

                hand_is_gesturing = True
                tip_x = int(landmarks[INDEX_TIP].x * width)
                tip_y = int(landmarks[INDEX_TIP].y * height)
                # Blue dot = pointing (volume), yellow dot = open hand (colors are BGR).
                dot_color = (255, 0, 0) if pointing else (0, 255, 255)
                cv2.circle(frame, (tip_x, tip_y), 15, dot_color, cv2.FILLED)

                direction = swipe_detector.update(tip_x, tip_y, now, vertical_only=pointing)
                if direction is None:
                    continue
                if not gestures_enabled:
                    action_label = f"(paused) ignored {direction} swipe"
                elif pointing:
                    action_label = adjust_volume(direction, volume_controller)
                elif direction in ("up", "down"):
                    action_label = scroll_tiktok(direction)
                else:
                    action_label = change_video_or_slide(direction)
                print(action_label)
                label_hide_time = time.time() + LABEL_DISPLAY_SEC

            # A pause in the gesture must not count as part of the next swipe.
            if not hand_is_gesturing:
                swipe_detector.reset()

            # ---- On-screen status ----
            cv2.putText(frame, f"Volume: {volume_controller.volume}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if now < label_hide_time:
                cv2.putText(frame, action_label, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if gestures_enabled:
                status, status_color = f"GESTURE CONTROL: ON   (press '{TOGGLE_KEY}' to pause)", (0, 255, 0)
            else:
                status, status_color = f"GESTURE CONTROL: PAUSED (press '{TOGGLE_KEY}' to resume)", (0, 0, 255)
            cv2.putText(frame, status, (10, height - 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, "Point + up/down = volume", (10, height - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"Open hand: up/down = scroll, left/right = {HORIZONTAL_SWIPE_TARGET}",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Unified Mac Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(TOGGLE_KEY):
                gestures_enabled = not gestures_enabled
                swipe_detector.reset()
                print(f"[toggle] Gesture control {'ENABLED' if gestures_enabled else 'PAUSED'}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()