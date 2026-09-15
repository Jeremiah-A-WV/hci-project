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

    frame = cv2.flip(frame, 1)
    
    # Process tracking safely on your Mac's CPU
    results = model(frame, device='cpu', verbose=False)
    
    # Draw the tracked hand points if detected
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.xy) > 0:
            # Flatten out the keypoint coordinate positions
            kp_data = result.keypoints.xy.cpu().numpy()[0]
            
            # Loop through all 21 points of the hand map
            for pt in kp_data:
                kx, ky = int(pt[0]), int(pt[1])
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (kx, ky), 5, (0, 255, 0), -1)

    cv2.imshow('YOLO Hand Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
