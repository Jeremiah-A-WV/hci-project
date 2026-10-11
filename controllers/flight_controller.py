from collections import deque
from codrone_edu.drone import Drone

class PositionSmoother:
    def __init__(self, size=5):
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
    def __init__(self, deadzone=15):
        self.drone = Drone()
        self.drone.pair()
        self.smoother = PositionSmoother()
        self.deadzone = deadzone
        self.is_flying = False

    def takeoff(self):
        self.drone.takeoff()
        self.is_flying = True

    def land(self):
        self.drone.land()
        self.is_flying = False

    def hover(self):
        self.smoother.reset()
        self.drone.hover()

    def navigate_by_coordinates(self, norm_x, norm_y):
        avg_x, avg_y = self.smoother.update(norm_x, norm_y)
        
        # Map normalized coordinates to drone power (-100 to 100)
        roll_power = int((avg_x - 0.5) * 200) 
        throttle_power = int((0.5 - avg_y) * 200) # Inverted because 0 is top of frame

        # Apply deadzone
        if abs(roll_power) < self.deadzone: roll_power = 0
        if abs(throttle_power) < self.deadzone: throttle_power = 0

        # Clamp values
        roll_power = max(-100, min(100, roll_power))
        throttle_power = max(-100, min(100, throttle_power))
        
        self.drone.set_roll(roll_power)
        self.drone.set_throttle(throttle_power)
        self.drone.move()
        
        return roll_power, throttle_power, avg_x, avg_y

    def safe_close(self):
        print("Cleaning up and closing drone connection...")
        self.drone.land()
        self.drone.close()