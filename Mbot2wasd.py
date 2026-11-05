"""Drive the mBot2 from the Mac keyboard over Bluetooth.

    W / S      forward / backward (hold to drive, let go to stop)
    A / D      turn left / right (alone = spin in place, with W or S = curve)
    Shift      boost
    Space      brake
    Q or Esc   quit (the robot stops)

Click the game window so it receives the keys.

One-time setup: install pygame and the mbot2 library (github.com/DrorSh/mbot_python).
Before running, turn the robot on and disconnect it in mBlock, because only one
program can hold its Bluetooth link.
"""
import os
import sys

# If the play button started a Python without these packages, restart this
# script with the project's virtual environment.
VENV_PYTHON = "/Users/gomezfile/Downloads/HCI_Work/hci-project/.venv/bin/python"
try:
    import pygame
    from mbot2 import MBot2
except ModuleNotFoundError:
    if os.path.exists(VENV_PYTHON) and not os.environ.get("HCI_RELAUNCHED"):
        os.environ["HCI_RELAUNCHED"] = "1"
        os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
    raise

# ---- Settings --------------------------------------------------------------
ROBOT_NAME = "Makeblock_LE08D1F96D2376"   # connect only to this robot, not a classmate's
DRIVE_SPEED = 40      # normal driving speed
BOOST_SPEED = 70      # driving speed while holding Shift
CURVE_AMOUNT = 25     # added to one wheel and removed from the other when curving
SPIN_SPEED = 35       # wheel speed when spinning in place
RIGHT_WHEEL_SIGN = -1 # the right motor is mirrored; flip this if W spins the robot
FRAMES_PER_SECOND = 30
# ---------------------------------------------------------------------------


def clamp_speed(speed):
    """Keeps a wheel speed within the robot's range of -100 to 100."""
    return max(-100, min(100, speed))


def wheel_speeds(forward, left, backward, right, boost=False, brake=False):
    """Turns the pressed keys into (left wheel, right wheel) speeds."""
    if brake:
        return 0, 0
    drive_direction = int(forward) - int(backward)   # +1 forward, -1 backward, 0 neither
    turn_direction = int(right) - int(left)          # +1 right, -1 left, 0 neither
    if drive_direction == 0:
        return turn_direction * SPIN_SPEED, -turn_direction * SPIN_SPEED
    speed = BOOST_SPEED if boost else DRIVE_SPEED
    left_speed = drive_direction * speed + turn_direction * CURVE_AMOUNT
    right_speed = drive_direction * speed - turn_direction * CURVE_AMOUNT
    return clamp_speed(left_speed), clamp_speed(right_speed)


def main():
    pygame.init()
    screen = pygame.display.set_mode((520, 300))
    pygame.display.set_caption("mBot2 WASD")
    title_font = pygame.font.SysFont(None, 30)
    body_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    def draw_screen(title, *body_lines):
        screen.fill((25, 28, 40))
        screen.blit(title_font.render(title, True, (255, 255, 255)), (20, 20))
        for line_number, line in enumerate(body_lines):
            screen.blit(body_font.render(line, True, (200, 200, 210)),
                        (20, 58 + 28 * line_number))
        pygame.display.flip()

    draw_screen("Connecting to mBot2 over Bluetooth...", "Robot on, disconnected from mBlock?")
    pygame.event.pump()   # lets macOS show the window before the connection wait
    print("Connecting over Bluetooth...")
    robot = MBot2(name=ROBOT_NAME)
    print("Connected.")

    last_speeds = None
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            keys = pygame.key.get_pressed()
            left_speed, right_speed = wheel_speeds(
                keys[pygame.K_w], keys[pygame.K_a], keys[pygame.K_s], keys[pygame.K_d],
                boost=keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
                brake=keys[pygame.K_SPACE])
            is_stopped = (left_speed, right_speed) == (0, 0)

            # Only message the robot when something changes, to keep Bluetooth quick.
            if (left_speed, right_speed) != last_speeds:
                if is_stopped:
                    robot.stop()
                else:
                    robot.drive(left_speed, RIGHT_WHEEL_SIGN * right_speed)
                last_speeds = (left_speed, right_speed)

            state = "STOPPED" if is_stopped else f"left {left_speed:+d}   right {right_speed:+d}"
            draw_screen(f"mBot2: {state}",
                        "W/S = forward/back    A/D = turn",
                        "Shift = boost    Space = brake",
                        "Q or Esc = quit",
                        "(click this window first so it gets the keys)")
            clock.tick(FRAMES_PER_SECOND)
    finally:
        print("Stopping and disconnecting...")
        try:
            robot.disconnect()   # also stops the robot
        finally:
            pygame.quit()


if __name__ == "__main__":
    main()