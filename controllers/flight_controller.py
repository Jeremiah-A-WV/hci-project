import time
import threading
import queue
from collections import deque
from codrone_edu.drone import Drone

class PIDController:
    """Calculates smoothed, dynamic flight outputs based on coordinate errors."""
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
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-4  # Prevent division by zero

        # Proportional
        p = self.kp * error

        # Integral
        self.integral += error * dt
        i = self.ki * self.integral

        # Derivative
        d = self.kd * (error - self.prev_error) / dt

        output = p + i + d

        # Clamp output to drone limits
        clamped_output = max(self.min_out, min(self.max_out, output))

        # Anti-Windup: Prevent the integral from endlessly building up if the 
        # output is already maxed out (e.g., holding your hand at the edge for too long)
        if clamped_output == self.max_out or clamped_output == self.min_out:
             self.integral -= error * dt 

        self.prev_error = error
        self.last_time = now

        return int(clamped_output)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()


class PositionSmoother:
    """Lightweight pre-filter to remove camera jitter before PID calculations."""
    def __init__(self, size=3):
        self.history_x = deque(maxlen=size)
        self.history_y = deque(maxlen=size)

    def update(self, x, y):
        self.history_x.append(x)
        self.history_y.append(y)
        return (sum(self.history_x) / len(self.history_x), 
                sum(self.history_y) / len(self.history_y))

    def reset(self):
        self.history_x.clear()
        self.history_y.clear()


class FlightController:
    def __init__(self, deadzone=0.08):
        self.drone = Drone()
        self.drone.pair()
        
        self.smoother = PositionSmoother()
        self.deadzone = deadzone
        self.is_flying = False

        # Tuning Knobs: You can adjust these based on how aggressive you want the drone to be.
        # Kp=150 scales the 0.0-0.5 max error close to ~75 power.
        # Kd=15 provides mild braking when moving the hand back to center.
        self.roll_pid = PIDController(kp=150, ki=5, kd=15)
        self.throttle_pid = PIDController(kp=150, ki=5, kd=15)

        self.command_queue = queue.Queue(maxsize=2)
        self.running = True
        
        self.worker = threading.Thread(target=self._flight_loop, daemon=True)
        self.worker.start()

    def _flight_loop(self):
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.1)
                action = cmd.get("action")
                
                if action == "takeoff":
                    self.drone.takeoff()
                    self.is_flying = True
                elif action == "land":
                    self.drone.land()
                    self.is_flying = False
                elif action == "hover":
                    self.smoother.reset()
                    self.roll_pid.reset()
                    self.throttle_pid.reset()
                    self.drone.hover()
                elif action == "move":
                    self.drone.set_roll(cmd["roll"])
                    self.drone.set_throttle(cmd["throttle"])
                    self.drone.move()
                    
                self.command_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Hardware Thread Error]: {e}")

    def _send_command(self, cmd_dict):
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
        self.command_queue.put(cmd_dict)

    def takeoff(self):
        self._send_command({"action": "takeoff"})

    def land(self):
        self._send_command({"action": "land"})

    def hover(self):
        self._send_command({"action": "hover"})

    def navigate_by_coordinates(self, norm_x, norm_y):
        avg_x, avg_y = self.smoother.update(norm_x, norm_y)
        
        # Calculate coordinate error from the center of the screen (0.5)
        roll_error = avg_x - 0.5 
        throttle_error = 0.5 - avg_y # Inverted: Y approaches 0 at the top of the screen
        
        # Apply deadzone (if the hand is very close to center, consider error 0)
        if abs(roll_error) < self.deadzone: roll_error = 0.0
        if abs(throttle_error) < self.deadzone: throttle_error = 0.0

        # Feed errors into the PID controllers
        roll_power = self.roll_pid.compute(roll_error)
        throttle_power = self.throttle_pid.compute(throttle_error)
        
        self._send_command({
            "action": "move",
            "roll": roll_power,
            "throttle": throttle_power
        })
        
        return roll_power, throttle_power, avg_x, avg_y

    def safe_close(self):
        print("Stopping hardware thread and closing connection...")
        self.running = False
        self.worker.join(timeout=2.0)
        self.drone.land()
        self.drone.close()