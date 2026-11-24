import time
import threading
import queue
import numpy as np
from collections import deque
from enum import Enum, auto
from codrone_edu.drone import Drone, Note

class FlightState(Enum):
    LANDED = auto()
    TAKING_OFF = auto()
    HOVERING = auto()
    NAVIGATING = auto()
    EMERGENCY = auto()

# note sequences as (Note, duration_ms)
SOUNDS = {
    "ready": [(Note.C4, 100), (Note.G4, 150)],
    "takeoff": [(Note.C4, 120), (Note.E4, 120), (Note.G4, 200)],
    "land": [(Note.G4, 120), (Note.E4, 120), (Note.C4, 250)],
    "flip": [(Note.E4, 80), (Note.G4, 80), (Note.C5, 180)],
    "drive_mode": [(Note.A4, 80), (Note.C5, 120)],
    "shield": [(Note.C5, 250)],
    "emergency": [(Note.C3, 200), (Note.C3, 350)],
}

class PIDController:
    def __init__(self, kp, ki, kd, max_out=100, min_out=-100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_out = min_out
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()

    def compute(self, error):
        # math magic for smooth flying
        now = time.time()
        dt = max(now - self.last_time, 1e-4)

        p = self.kp * error
        self.integral += error * dt
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt

        output = max(self.min_out, min(self.max_out, p + i + d))

        # stop integral windup
        if output in (self.max_out, self.min_out):
            self.integral -= error * dt

        self.prev_error = error
        self.last_time = now
        return int(output)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()

class PredictiveSmoother:
    # guesses where your hand is going to fight lag
    def __init__(self, history_size=8, lookahead_frames=3):
        self.history_x = deque(maxlen=history_size)
        self.history_y = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.lookahead = lookahead_frames

    def update(self, x, y):
        now = time.time()
        self.history_x.append(x)
        self.history_y.append(y)
        self.timestamps.append(now)

        if len(self.history_x) < 3:
            return x, y

        t = np.array(self.timestamps) - self.timestamps[0]
        m_x, b_x = np.polyfit(t, self.history_x, 1)
        m_y, b_y = np.polyfit(t, self.history_y, 1)

        avg_dt = (self.timestamps[-1] - self.timestamps[0]) / len(self.timestamps)
        future_t = t[-1] + (avg_dt * self.lookahead)

        return (m_x * future_t + b_x), (m_y * future_t + b_y)

    def reset(self):
        self.history_x.clear()
        self.history_y.clear()
        self.timestamps.clear()

class FlightController:
    def __init__(self, deadzone=0.08):
        self.drone = Drone()
        self.drone.pair()

        self.smoother = PredictiveSmoother()
        self.deadzone = deadzone
        self.state = FlightState.LANDED

        # pids for 1-hand camera tracking
        self.roll_pid = PIDController(150, 5, 15)
        self.throttle_pid = PIDController(150, 5, 15)

        # full keyboard steering states
        self.keyboard_pitch = 0
        self.keyboard_yaw = 0
        self.keyboard_roll = 0
        self.keyboard_throttle = 0
        self.battery = 100

        # shield tracker + bluetooth rate-limit cache
        self.is_braking = False
        self.cached_front_dist = 999
        self.last_sensor_check = 0

        # stop buzzer melodies from talking over each other
        self._buzzer_lock = threading.Lock()

        self.command_queue = queue.Queue(maxsize=2)
        self.running = True

        # boot up the background worker
        self.worker = threading.Thread(target=self._flight_loop, daemon=True)
        self.worker.start()

        self.play_sound("ready")

    def play_sound(self, sound_name, force=False):
        # fire-and-forget melody thread so motors never stall
        notes = SOUNDS.get(sound_name)
        if not notes:
            return

        def _play():
            if not self._buzzer_lock.acquire(blocking=force):
                return
            try:
                for note, duration in notes:
                    self.drone.drone_buzzer(note, duration)
                    time.sleep(0.03) # tiny gap so notes stay crisp
            except Exception:
                pass
            finally:
                self._buzzer_lock.release()

        threading.Thread(target=_play, daemon=True).start()

    def _flight_loop(self):
        last_battery = 0
        while self.running:
            try:
                now = time.time()
                if now - last_battery > 10:
                    self.battery = self.drone.get_battery()
                    last_battery = now

                cmd = self.command_queue.get(timeout=0.1)
                action = cmd.get("action")

                if action == "takeoff":
                    self.play_sound("takeoff")
                    self.drone.takeoff()
                    if self.state != FlightState.EMERGENCY:
                        self.state = FlightState.HOVERING
                        self.drone.set_drone_LED(0, 255, 0, 100) # start green
                elif action == "land":
                    self.is_braking = False
                    self.cached_front_dist = 999
                    self.play_sound("land")
                    self.drone.land()
                    if self.state != FlightState.EMERGENCY:
                        self.state = FlightState.LANDED
                elif action == "hover":
                    self.smoother.reset()
                    self.roll_pid.reset()
                    self.throttle_pid.reset()
                    if self.is_braking:
                        self.drone.set_drone_LED(0, 255, 0, 100)
                        self.is_braking = False
                        self.cached_front_dist = 999
                    self.drone.hover()
                elif action == "move":
                    pitch = cmd["pitch"]

                    # invisible shield (rate-limited so bluetooth doesn't choke)
                    if pitch > 0:
                        if now - self.last_sensor_check > 0.15:
                            try:
                                dist = self.drone.get_front_range()
                                if dist is not None and dist > 0:
                                    self.cached_front_dist = dist
                            except Exception:
                                pass # shrug off dropped packets
                            self.last_sensor_check = now

                        if self.cached_front_dist < 30:
                            pitch = 0 # override forward flight
                            if not self.is_braking:
                                self.drone.set_drone_LED(255, 0, 0, 100) # flash red
                                self.play_sound("shield")
                                self.is_braking = True
                        else:
                            if self.is_braking:
                                self.drone.set_drone_LED(0, 255, 0, 100) # back to green
                                self.is_braking = False
                    else:
                        if self.is_braking:
                            self.drone.set_drone_LED(0, 255, 0, 100)
                            self.is_braking = False
                            self.cached_front_dist = 999

                    self.drone.set_roll(cmd["roll"])
                    self.drone.set_pitch(pitch)
                    self.drone.set_yaw(cmd["yaw"])
                    self.drone.set_throttle(cmd["throttle"])
                    self.drone.move()
                elif action == "flip":
                    # hop-and-flip for tighter rolls
                    self.state = FlightState.NAVIGATING
                    self.play_sound("flip")
                    self.drone.set_pitch(0)
                    self.drone.set_roll(0)
                    self.drone.set_yaw(0)
                    self.drone.set_throttle(70)
                    self.drone.move()
                    time.sleep(0.2)
                    self.drone.flip(cmd.get("direction", "l"))
                    if self.state != FlightState.EMERGENCY:
                        self.state = FlightState.HOVERING

                self.command_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                print(f"yikes, hardware error: {e}")

    def _send_command(self, cmd_dict):
        # keep queue from backing up
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
        self.command_queue.put(cmd_dict)

    def takeoff(self):
        if self.state == FlightState.LANDED:
            self.state = FlightState.TAKING_OFF
            self._send_command({"action": "takeoff"})

    def land(self):
        if self.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
            self.state = FlightState.LANDED
            self._send_command({"action": "land"})

    def hover(self):
        if self.state == FlightState.NAVIGATING:
            self.state = FlightState.HOVERING
            self._send_command({"action": "hover"})

    def emergency_stop(self):
        # cut motors first, buzz second
        if self.state != FlightState.EMERGENCY:
            self.state = FlightState.EMERGENCY
            self.is_braking = False
            with self.command_queue.mutex:
                self.command_queue.queue.clear()
            self.drone.emergency_stop()
            self.play_sound("emergency", force=True)

    def flip(self, direction="l"):
        if self.state in [FlightState.HOVERING, FlightState.NAVIGATING]:
            self._send_command({"action": "flip", "direction": direction})

    def update_keyboard_steering(self, pitch, yaw, roll=0, throttle=0):
        self.keyboard_pitch = pitch
        self.keyboard_yaw = yaw
        self.keyboard_roll = roll
        self.keyboard_throttle = throttle

    def navigate_by_coordinates(self, norm_x, norm_y):
        # 1-hand finger tracking
        if self.state not in [FlightState.HOVERING, FlightState.NAVIGATING]:
            return 0, 0, norm_x, norm_y

        self.state = FlightState.NAVIGATING
        pred_x, pred_y = self.smoother.update(norm_x, norm_y)

        err_x = pred_x - 0.5
        err_y = 0.5 - pred_y

        if abs(err_x) < self.deadzone: err_x = 0.0
        if abs(err_y) < self.deadzone: err_y = 0.0

        roll = self.roll_pid.compute(err_x)
        throttle = self.throttle_pid.compute(err_y)

        self._send_command({
            "action": "move",
            "roll": roll,
            "pitch": self.keyboard_pitch,
            "yaw": self.keyboard_yaw,
            "throttle": throttle
        })

        return roll, throttle, pred_x, pred_y

    def drive_vehicle(self, steering_angle, gas_input):
        # 2-hand virtual steering wheel
        if self.state not in [FlightState.HOVERING, FlightState.NAVIGATING]:
            return 0, 0

        self.state = FlightState.NAVIGATING

        # vertical wheel height -> gas / reverse
        pitch_power = int(gas_input * 250)
        if abs(pitch_power) < 15:
            pitch_power = 0
        pitch_power = max(-45, min(45, pitch_power))

        # wheel tilt -> turning
        yaw_power = int(steering_angle * 1.5)
        if abs(yaw_power) < 10:
            yaw_power = 0
        yaw_power = max(-60, min(60, yaw_power))

        self._send_command({
            "action": "move",
            "roll": 0,
            "pitch": pitch_power,
            "yaw": yaw_power,
            "throttle": 0
        })

        return yaw_power, pitch_power

    def safe_close(self):
        self.running = False
        self.worker.join(timeout=2.0)
        self.drone.land()
        self.drone.close()