import time
import pydirectinput


while True:
    time.sleep(0.5)
    current_x, current_y = pydirectinput.position()
    print(f"current_x:{current_x}, current_y:{current_y}")