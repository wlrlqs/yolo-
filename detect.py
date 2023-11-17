import ctypes
import threading
import time
from math import sqrt
import multiprocessing
import pydirectinput
import numpy as np
import mss
import PIL.Image
import cv2
import win32con
import win32gui
import win32ui
from ultralytics import YOLO
from multiprocessing import Queue
import keyboard

import logitech_test

# xywh = [0, 0, 2560, 1440]
# xywh = [0, 0, 1280, 720]
Kp = 0.2
Ki = 0.001
Kd = 0.2
# 初始误差、累积量和上次误差
x_error = 0
x_integral1 = 0
x_previous_error = 0
x_error = 0
x_previous_error = 0
x_error = 0
x_derivative = 0
y_error = 0
y_integral1 = 0
y_previous_error = 0
y_error = 0
y_previous_error = 0
y_error = 0
y_derivative = 0

x_range = 2560
y_range = 1440
get_x_range = int(512)
get_y_range = int(288)  # int(get_x_range * y_range / x_range)
x_start = int((x_range - get_x_range) / 2)
y_start = int((y_range - get_y_range) / 2)
xywh = [x_start, y_start, get_x_range, get_y_range]
screen_center = [get_x_range / 2, get_y_range / 2]
sct = mss.mss()
monitor = {"top": y_start, "left": x_start, "width": get_x_range, "height": get_y_range}
hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, xywh[2], xywh[3])
memdc.SelectObject(bmp)

shared_memory = multiprocessing.RawArray(ctypes.c_ubyte, get_y_range * get_x_range * 3)


def get_screen_no():
    timer1 = time.time()
    img = np.array(sct.grab(monitor))
    # 转换为PIL图像
    # img_pil = PIL.Image.fromarray(img)
    img.shape = (get_y_range, get_x_range, 4)
    frame = cv2.resize((img[:, :, :-1]), (get_x_range, get_y_range))
    timer2 = time.time()
    print(f"grab_screen fps:{1 / (timer2 - timer1)}")
    return frame


def get_screen(shared_memory,lock1):
    while True:
        timer1 = time.time()
        img = np.array(sct.grab(monitor))
        # 转换为PIL图像
        # img_pil = PIL.Image.fromarray(img)
        img.shape = (get_y_range, get_x_range, 4)
        # frame = cv2.resize((img[:, :, :-1]), (get_x_range, get_y_range))
        # np.copyto(shared_memory, frame)
        frame = cv2.resize((img[:, :, :-1]), (get_x_range, get_y_range))
        shared_memory_ptr = (ctypes.c_ubyte * (get_x_range * get_y_range * 3)).from_address(
            ctypes.addressof(shared_memory))
        with lock1:
            ctypes.memmove(shared_memory_ptr, frame.tobytes(), get_x_range * get_y_range * 3)
            timer2 = time.time()
            print(f"grab_screen_time:{timer2 - timer1}")
        # grapQueue.put(frame)


def grab_screen(grapQueue):
    while True:
        timer1 = time.time()
        memdc.BitBlt((0, 0), (xywh[2], xywh[3]), srcdc, (xywh[0], xywh[1]), win32con.SRCCOPY)
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, np.uint8)
        img.shape = (xywh[3], xywh[2], 4)
        frame = cv2.resize(img[:, :, :-1], (get_x_range, get_y_range))
        grapQueue.put(frame)
        timer2 = time.time()
        print(f"grab_screen fps:{1 / (timer2 - timer1)}")


def decide(result, lock):
        distList = []

        for item in result.boxes.xywh:
            center = (float(item[0]), float(item[1]))
            dist = sqrt(((center[0] - screen_center[0]) ** 2 + (center[1] - screen_center[1]) ** 2))
            # print(dist)
            distList.append(dist)
        if not len(distList):
            return
        index = distList.index(min(distList))
        targetBox = result.boxes.xywh[index]
        targetPoint = [int(targetBox[0] + x_start), int(targetBox[1] + y_start)]
        targetMove = [int(x - y) for x, y in zip(targetPoint, screen_center)]
        current_x, current_y = pydirectinput.position()
        current_x = current_x
        current_y = current_y
        targetMove = [int(targetPoint[0] - current_x), int(targetPoint[1] - current_y)]
        print(targetPoint)
        # print(f"screen_center = {screen_center} distList = {distList}")
        # print(f"targetPointX:{targetPoint[0]} targetPointY:{targetPoint[1]}")
        # print(f"targetMoveX:{targetMove[0]} targetMoveY:{targetMove[1]}")
        # 移动鼠标到指定点
        # pydirectinput.moveTo(int(targetPoint[0] / 1.25), int(targetPoint[1] / 1.25))
        # pydirectinput.moveTo(int(targetPoint[0] / 1.25), int(targetPoint[1] / 1.25), duration=0.1, relative=True)
        # pydirectinput.moveRel(int(targetPoint[0] / 1.25), int(targetPoint[1] / 1.25),duration=0.4,relative=True)
        # pydirectinput.PAUSE = 0.0

        with lock:
            target[0] = targetPoint[0]
            target[1] = targetPoint[1]
        # mouseQueue.put(targetMove)
        # pydirectinput.moveRel(int(targetMove[0] / 1.25), int(targetMove[1] / 1.25), duration=0.4, relative=True)
        # if abs(targetMove[0]) + abs(targetMove[1]) < 15:
        #     logitech_test.Logitech.mouse.press(1)
        # logitech_test.Logitech.mouse.move(int(targetMove[0] / 1.25 / 5), int(targetMove[1] / 1.25 / 5))
        # if keyboard.is_pressed('1'):
        logitech_test.Logitech.mouse.move(int(targetMove[0] / 1.25), int(targetMove[1] / 1.25))
        time.sleep(0.005)
        logitech_test.Logitech.mouse.move(int(0), int(0))
        if abs(targetMove[0]) + abs(targetMove[1]) < 15:
            logitech_test.Logitech.mouse.press(1)
        if keyboard.is_pressed('8'):
            return



def mouse_control(mouseQueue, lock):
    global x_integral1, x_previous_error, x_error, x_derivative, y_integral1, y_previous_error, y_error, Kp, Kd
    targetPoint = [0, 0]
    while True:
        with lock:
            targetPoint[0] = mouseQueue[0]
            targetPoint[1] = mouseQueue[1]
        # print(targetPoint)
        targetMove = [int(x - y) for x, y in zip(targetPoint, screen_center)]
        current_x, current_y = pydirectinput.position()
        current_x = current_x
        current_y = current_y
        targetMove = [int(targetPoint[0] - current_x), int(targetPoint[1] - current_y)]
        if keyboard.is_pressed('2'):
            # if abs(targetMove[0]) + abs(targetMove[1]) < 15:
            #     logitech_test.Logitech.mouse.press(1)
            center = (float(targetPoint[0]), float(targetPoint[1]))
            dist = sqrt(((center[0] - screen_center[0]) ** 2 + (center[1] - screen_center[1]) ** 2))
            fbd = pydirectinput.position()

            x_error = targetPoint[0] - fbd[0]
            x_integral1 += x_error
            if abs(x_error) < 100:
                x_integral1 = 0
            x_derivative = x_error - x_previous_error
            xoutput = Kp * x_error + Ki * x_integral1 + Kd * x_derivative
            x_previous_error = x_error

            y_error = targetPoint[1] - fbd[1]
            y_integral1 += y_error
            if abs(y_error) < 100:
                y_integral1 = 0
            y_derivative = y_error - y_previous_error
            youtput = Kp * y_error + Ki * y_integral1 + Kd * y_derivative
            y_previous_error = y_error


            logitech_test.Logitech.mouse.move(int(xoutput), int(youtput))
            time.sleep(0.0005)
            logitech_test.Logitech.mouse.move(int(0), int(0))
        else:
            x_integral1 = 0
            y_integral1 = 0
        # if keyboard.is_pressed('1'):
        # print(f"targetPoint in mouse{targetPoint}")
        # print(f"current_x in mouse{current_x}")
        # print(f"current_y in mouse{current_y}")
        # print(f"pydirectinput.position():{pydirectinput.position()}")
        # print(f"targetMove in mouse{targetMove}")

timer0 = 1
if __name__ == "__main__":
    grapQueue = Queue(1)
    manager = multiprocessing.Manager()
    target = manager.list([0, 0, 0])
    lock = multiprocessing.Lock()  # 创建一个锁对象
    lock1 = multiprocessing.Lock()  # 创建一个锁对象
    grapQueue.put((np.zeros((10, 10, 3), dtype=np.uint8), time.time()))
    grab = multiprocessing.Process(target=get_screen, args=(shared_memory,lock1))
    mouse = multiprocessing.Process(target=mouse_control, args=(target, lock))
    mouse.start()
    grab.start()
    model = YOLO('best-n.pt')
    frame = get_screen_no()
    result = model.predict(frame, show=False, conf=0.5)[0]
    i = 0
    while True:
        print(f"while_time:{time.time() - timer0}")
        # frame = grapQueue.get()
        # frame1 = get_screen_no()
        timer0 = time.time()
        with lock1:
            main_img_array = np.frombuffer(shared_memory, dtype=np.uint8)
        frame = main_img_array.reshape(get_y_range, get_x_range, 3)
        print(f"lock_wait_time:{time.time() - timer0}")
        timer1 = time.time()
        result = model.predict(frame, show=False, conf=0.5)[0]
        print(f"modle_time:{time.time() - timer1}")
        timer2 = time.time()
        decide(result, lock)
        # t = threading.Thread(target=decide, args=(result, lock))
        # t.start()
        timer3 = time.time()
        cv2.imshow("tragetImg", cv2.resize(result.plot(), [get_x_range*2, get_y_range*2]))
        key = cv2.waitKey(1)
        print(f"decide_time:{timer3 - timer2}")
        print(f"show_time:{time.time() - timer3}")
        # i += 1
        # while_time += time.time() - timer0
        # if i == 50:
        #     print(f"modle_time:{modle_time / 50}")
        #     print(f"while_time:{while_time / 50}")
        #     print(f"while fps:{1 / (while_time / 50)}")
        #     modle_time = 0
        #     while_time = 0
        #     i = 0
# model = YOLO(r'C:\Users\Lanye\Desktop\Code\ubuntu\runs\detect\train9\weights\best.pt')
# cap = cv2.VideoCapture(r"C:\Users\Lanye\Desktop\cod.mp4")
# while True:
#     ret, frame = cap.read()
#     if ret:
#         result = model.predict(frame, show=True, conf=0.5)
#         print([res.boxes for res in result])
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
