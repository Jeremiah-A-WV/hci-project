import os
import cv2
import time
import math
import asyncio
from contextlib import asynccontextmanager

# grab the real sleep before codrone breaks it
_original_asyncio_sleep = asyncio.sleep

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from detectors.gesture_detector import GestureDetector, INDEX_TIP
from controllers.flight_controller import FlightController, FlightState

# put the real sleep back
asyncio.sleep = _original_asyncio_sleep

# block annoying mac background features
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CMIO_DISABLE_PORTRAIT_EFFECTS"] = "1"

drone = FlightController(deadzone=0.08)
detector = GestureDetector()
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    drone.safe_close()
    cap.release()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# what we send to react
telemetry_data = {
    "status": "LANDED",
    "roll": 0,
    "throttle": 0,
    "pitch": 0,
    "yaw": 0,
    "battery": 100,
    "shield_active": False
}

def process_video_frames():
    global telemetry_data
    last_gesture_time = 0.0
    in_driving_mode = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1) # mirror it so it's not confusing
        h, w, _ = frame.shape
        now = time.time()

        telemetry_data["battery"] = drone.battery
        telemetry_data["status"] = drone.state.name
        telemetry_data["shield_active"] = drone.is_braking

        # mediapipe stuff
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = detector.process_frame(rgb_frame)
        rgb_frame.flags.writeable = True

        detected_hands = results.multi_hand_landmarks or []
        for hand_lms in detected_hands:
            detector.drawing_utils.draw_landmarks(
                frame, hand_lms, detector.connections,
                detector.drawing_styles.get_default_hand_landmarks_style(),
                detector.drawing_styles.get_default_hand_connections_style()
            )

        tracking = False

        # 2 hands up -> virtual steering wheel mode
        if len(detected_hands) == 2 and drone.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
            tracking = True

            # chirp once when locking into steering wheel mode
            if not in_driving_mode:
                drone.play_sound("drive_mode")
                in_driving_mode = True

            hand1 = detected_hands[0].landmark[0] # wrist
            hand2 = detected_hands[1].landmark[0] # wrist
            left_hand, right_hand = (hand1, hand2) if hand1.x < hand2.x else (hand2, hand1)

            # wheel angle + vertical height
            angle_deg = math.degrees(math.atan2(right_hand.y - left_hand.y, right_hand.x - left_hand.x))
            mid_y = (left_hand.y + right_hand.y) / 2
            gas_input = 0.5 - mid_y

            yaw, pitch = drone.drive_vehicle(angle_deg, gas_input)

            # draw virtual steering wheel on HUD
            left_px = (int(left_hand.x * w), int(left_hand.y * h))
            right_px = (int(right_hand.x * w), int(right_hand.y * h))
            mid_px = (int((left_px[0] + right_px[0]) / 2), int((left_px[1] + right_px[1]) / 2))

            cv2.line(frame, left_px, right_px, (0, 255, 255), 4)
            cv2.line(frame, mid_px, (mid_px[0], int(h * 0.5)), (255, 0, 255), 4)
            cv2.circle(frame, left_px, 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, right_px, 15, (0, 0, 255), cv2.FILLED)

            telemetry_data["yaw"] = yaw
            telemetry_data["pitch"] = pitch
            telemetry_data["roll"] = 0
            telemetry_data["throttle"] = 0
            telemetry_data["status"] = "DRIVING"

        # 1 hand up -> takeoff / land / finger tracking
        elif len(detected_hands) == 1:
            in_driving_mode = False
            hand_lms = detected_hands[0].landmark
            is_pointing = detector.is_index_pointing(hand_lms)
            is_open = detector.is_whole_hand_open(hand_lms)
            is_peace = detector.is_peace_sign(hand_lms)

            if is_open and drone.state == FlightState.LANDED:
                if now - last_gesture_time > 2.0:
                    drone.takeoff()
                    last_gesture_time = now

            elif is_peace and drone.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
                if now - last_gesture_time > 2.0:
                    drone.land()
                    last_gesture_time = now

            elif is_pointing and drone.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
                tracking = True
                tip = hand_lms[INDEX_TIP]

                roll, throttle, pred_x, pred_y = drone.navigate_by_coordinates(tip.x, tip.y)

                # draw targeting visuals
                raw_px = (int(tip.x * w), int(tip.y * h))
                pred_px = (int(pred_x * w), int(pred_y * h))

                cv2.circle(frame, raw_px, 10, (255, 255, 0), cv2.FILLED)
                cv2.circle(frame, pred_px, 15, (0, 0, 255), 2)
                cv2.line(frame, raw_px, pred_px, (0, 255, 0), 2)

                telemetry_data["roll"] = roll
                telemetry_data["throttle"] = throttle
        else:
            in_driving_mode = False

        # check if user is flying with keyboard keys
        is_keyboard_driving = (
            drone.keyboard_pitch != 0 or
            drone.keyboard_yaw != 0 or
            drone.keyboard_roll != 0 or
            drone.keyboard_throttle != 0
        )

        # auto-hover only if no hand tracking and no keys held
        if not tracking and not is_keyboard_driving and drone.state == FlightState.NAVIGATING:
            drone.hover()
            telemetry_data["roll"] = 0
            telemetry_data["throttle"] = 0
            telemetry_data["pitch"] = 0
            telemetry_data["yaw"] = 0

        # keyboard-only flight when hands are down
        elif not tracking and is_keyboard_driving and drone.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
            drone.state = FlightState.NAVIGATING
            drone._send_command({
                "action": "move",
                "roll": drone.keyboard_roll,
                "pitch": drone.keyboard_pitch,
                "yaw": drone.keyboard_yaw,
                "throttle": drone.keyboard_throttle
            })

        # zero out graphs when grounded
        if drone.state in [FlightState.LANDED, FlightState.EMERGENCY]:
            in_driving_mode = False
            telemetry_data["roll"] = 0
            telemetry_data["throttle"] = 0
            telemetry_data["pitch"] = 0
            telemetry_data["yaw"] = 0

        # ship it to the web
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(process_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def listen_for_commands():
        try:
            while True:
                data = await websocket.receive_json()
                cmd = data.get("command")

                # react to keyboard stuff
                if cmd == "emergency_stop":
                    drone.emergency_stop()
                elif cmd == "flip":
                    drone.flip("l") # barrel roll
                elif cmd == "takeoff":
                    drone.takeoff()
                elif cmd == "land":
                    drone.land()
                elif cmd == "steer":
                    pitch = data.get("pitch", 0)
                    yaw = data.get("yaw", 0)
                    roll = data.get("roll", 0)
                    throttle = data.get("throttle", 0)
                    drone.update_keyboard_steering(pitch, yaw, roll, throttle)
                    telemetry_data["pitch"] = pitch
                    telemetry_data["yaw"] = yaw
                    telemetry_data["roll"] = roll
                    telemetry_data["throttle"] = throttle
        except Exception:
            pass

    listen_task = asyncio.create_task(listen_for_commands())

    try:
        while True:
            await websocket.send_json(telemetry_data)
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"ws disconnected: {e}")
    finally:
        listen_task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)