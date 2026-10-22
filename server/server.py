import os
import cv2
import time
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from detectors.gesture_detector import GestureDetector, INDEX_TIP
from controllers.flight_controller import FlightController

# Forcefully block macOS from injecting background camera portrait/blur layers
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
drone = FlightController(deadzone=0.08)
detector = GestureDetector()
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Shared telemetry state
telemetry_data = {
    "status": "Landed",
    "roll": 0,
    "throttle": 0,
    "battery": 0
}

GESTURE_COOLDOWN_SEC = 2.0
last_gesture_time = 0.0

def process_video_frames():
    """Generator that runs the computer vision loop and yields JPEG frames."""
    global telemetry_data, last_gesture_time
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
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
                # Draw skeleton
                detector.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, detector.connections,
                    detector.drawing_styles.get_default_hand_landmarks_style(),
                    detector.drawing_styles.get_default_hand_connections_style()
                )

                is_pointing = detector.is_index_pointing(hand_landmarks.landmark)
                is_open = detector.is_whole_hand_open(hand_landmarks.landmark)
                is_peace = detector.is_peace_sign(hand_landmarks.landmark)

                if is_open and not drone.is_flying:
                    if now - last_gesture_time > GESTURE_COOLDOWN_SEC:
                        telemetry_data["status"] = "Taking Off"
                        drone.takeoff()
                        last_gesture_time = now
                        
                elif is_peace and drone.is_flying:
                    if now - last_gesture_time > GESTURE_COOLDOWN_SEC:
                        telemetry_data["status"] = "Landing"
                        drone.land()
                        last_gesture_time = now

                elif is_pointing and drone.is_flying:
                    tracking_this_frame = True
                    tip = hand_landmarks.landmark[INDEX_TIP]
                    roll, throttle, avg_x, avg_y = drone.navigate_by_coordinates(tip.x, tip.y)
                    
                    cv2.circle(frame, (int(avg_x * w), int(avg_y * h)), 15, (255, 0, 0), cv2.FILLED)
                    
                    telemetry_data["status"] = "Navigating"
                    telemetry_data["roll"] = roll
                    telemetry_data["throttle"] = throttle

        if not tracking_this_frame and drone.is_flying:
            drone.hover()
            telemetry_data["status"] = "Hovering"
            telemetry_data["roll"] = 0
            telemetry_data["throttle"] = 0
        elif not drone.is_flying:
            telemetry_data["status"] = "Landed"
            telemetry_data["roll"] = 0
            telemetry_data["throttle"] = 0

        # Encode frame as JPEG for the web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """Endpoint for the React <img> tag to consume the MJPEG stream."""
    return StreamingResponse(process_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """Pushes live drone telemetry to the React frontend."""
    await websocket.accept()
    try:
        while True:
            # Optionally grab real battery if drone supports it non-blockingly
            # telemetry_data["battery"] = drone.drone.get_battery() 
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.1)  # 10Hz refresh rate
    except Exception as e:
        print(f"WebSocket disconnected: {e}")

@app.on_event("shutdown")
def shutdown_event():
    drone.safe_close()
    cap.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)