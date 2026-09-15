import cv2
from ultralytics import YOLO

# Load the specialized hand keypoint model directly via the framework
# This uses the exact 21-point hand mapping setup natively
model = YOLO("best.pt") 

# Direct targeting of your Mac FaceTime camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # frame = cv2.flip(frame, 1)
    
    # Process tracking safely on your Mac's CPU
    results = model(frame, device='cpu', verbose=False)

    # Standard 21-point hand skeleton connections
    HAND_SKELETON = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # Index
        (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)    # Pinky
    ]
    
    # Draw the tracked hand points if detected
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.xy) > 0:
            
            # Iterate over ALL detected hands in the frame
            for hand_kpts, hand_confs in zip(result.keypoints.xy, result.keypoints.conf):
                kpts = hand_kpts.cpu().numpy()
                confs = hand_confs.cpu().numpy()
                
                # 1. Draw the skeletal connections
                for start_idx, end_idx in HAND_SKELETON:
                    # Only draw the line if BOTH joints are above 50% confidence
                    if confs[start_idx] > 0.5 and confs[end_idx] > 0.5:
                        pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                        pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue lines

                # 2. Draw the joints (dots) over the lines
                for (x, y), conf in zip(kpts, confs):
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('YOLO Hand Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()