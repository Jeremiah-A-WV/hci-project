import os
import time
import cv2

from detectors.gesture_detector import GestureDetector, INDEX_TIP
from controllers.flight_controller import FlightController

# Forcefully block macOS from injecting background camera portrait/blur layers
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

GESTURE_COOLDOWN_SEC = 2.0

def main():
    # Initialize the separated modules
    drone = FlightController(deadzone=15)
    detector = GestureDetector()
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    last_gesture_time = 0.0
    status_label = "Landed (Open hand to takeoff)"

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = detector.process_frame(rgb_frame)
            rgb_frame.flags.writeable = True

            tracking_this_frame = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    # Draw visual skeleton
                    detector.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        detector.connections,
                        detector.drawing_styles.get_default_hand_landmarks_style(),
                        detector.drawing_styles.get_default_hand_connections_style()
                    )

                    # Evaluate hand states
                    is_pointing = detector.is_index_pointing(hand_landmarks.landmark)
                    is_open = detector.is_whole_hand_open(hand_landmarks.landmark)
                    is_peace = detector.is_peace_sign(hand_landmarks.landmark)

                    # Takeoff Trigger
                    if is_open and not drone.is_flying:
                        if now - last_gesture_time > GESTURE_COOLDOWN_SEC:
                            status_label = "Taking Off!"
                            print(status_label)
                            drone.takeoff()
                            last_gesture_time = now
                            
                    # Land Trigger
                    elif is_peace and drone.is_flying:
                        if now - last_gesture_time > GESTURE_COOLDOWN_SEC:
                            status_label = "Landing..."
                            print(status_label)
                            drone.land()
                            last_gesture_time = now

                    # Continuous Flight Control (When pointing index finger)
                    elif is_pointing and drone.is_flying:
                        tracking_this_frame = True
                        tip = hand_landmarks.landmark[INDEX_TIP]
                        
                        # Pass normalized coordinates to the flight controller
                        roll, throttle, avg_x, avg_y = drone.navigate_by_coordinates(tip.x, tip.y)
                        
                        # Visual feedback marker for the smoothed coordinate
                        cv2.circle(frame, (int(avg_x * w), int(avg_y * h)), 15, (255, 0, 0), cv2.FILLED)
                        status_label = f"Flying | Roll: {roll} Thr: {throttle}"

            # Safely hover if tracking is lost mid-flight
            if not tracking_this_frame and drone.is_flying:
                drone.hover()
                status_label = "Hovering (Point finger to move)"
            elif not drone.is_flying:
                status_label = "Landed (Open hand to takeoff)"

            # Render HUD
            cv2.putText(frame, status_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('CoDrone EDU Gesture Controller', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        drone.safe_close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()