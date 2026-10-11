import mediapipe as mp

# MediaPipe Hands topology indices
WRIST = 0
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        
            max_num_hands=1,
            model_complexity=1,             
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.connections = self.mp_hands.HAND_CONNECTIONS
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, rgb_frame):
        return self.hands.process(rgb_frame)

    @staticmethod
    def is_index_pointing(landmarks):
        """True when the index finger is extended and the other fingers are curled."""
        index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
        middle_curled = landmarks[MIDDLE_TIP].y > landmarks[MIDDLE_PIP].y
        ring_curled = landmarks[RING_TIP].y > landmarks[RING_PIP].y
        pinky_curled = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
        return index_extended and middle_curled and ring_curled and pinky_curled

    @staticmethod
    def is_whole_hand_open(landmarks):
        """True when the index, middle, ring, and pinky fingers are all extended."""
        index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
        middle_extended = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
        ring_extended = landmarks[RING_TIP].y < landmarks[RING_PIP].y
        pinky_extended = landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y
        return index_extended and middle_extended and ring_extended and pinky_extended

    @staticmethod
    def is_peace_sign(landmarks):
        """True when the index and middle fingers are extended, and ring/pinky are curled."""
        index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
        middle_extended = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
        ring_curled = landmarks[RING_TIP].y > landmarks[RING_PIP].y
        pinky_curled = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
        return index_extended and middle_extended and ring_curled and pinky_curled