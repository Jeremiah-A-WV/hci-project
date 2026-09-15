import os

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

# 3. Target your Mac FaceTime camera using AVFOUNDATION backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

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

        # Convert OpenCV BGR frame to MediaPipe required RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performance optimization: Mark image as flag non-writeable to speed up pass-by-reference
        rgb_frame.flags.writeable = False
        
        # Process the frame (Legacy infers timestamp tracking out of the box)
        results = hands.process(rgb_frame)

        # Draw overlays if hands are tracked
        rgb_frame.flags.writeable = True
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
                wrist_landmark = hand_landmarks.landmark[0]
                cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
                cv2.putText(frame, hand_label, (cx - 20, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display window
        cv2.imshow('MediaPipe Legacy Hand Tracker', frame)

        # Press 'q' to exit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
