import cv2
from ultralytics import YOLO
import time
model = YOLO('best-n.pt')
i = 0
frame = cv2.imread(r"E:\dataset_video\Call of Duty  Modern Warfare 2 (2022)\photo3\training_ground_411.jpg")
frame = cv2.resize(frame,(1024,576))
while True:
    timer1 = time.time()
    result = model.predict(frame, show=False, conf=0.5)[0]
    i += 1
    if i == 101 :
        timer2 = time.time()
        print(f"fps = {1 /(timer2 - timer1)}")
        i = 0
        timer1 = time.time()