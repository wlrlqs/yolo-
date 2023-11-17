import numpy as np
import mss
import PIL.Image
import time

# 创建 mss 实例
sct = mss.mss()

# 设置截屏区域的左上角坐标和宽度、高度
monitor = {"top": 0, "left": 0, "width": 1024, "height": 576}
def get_screen():
    img = np.array(sct.grab(monitor))
    # 转换为PIL图像
    img_pil = PIL.Image.fromarray(img)

while True:
    # 记录开始时间
    start_time = time.time()

    # 获取屏幕截图
    img = np.array(sct.grab(monitor))

    # 转换为PIL图像
    img_pil = PIL.Image.fromarray(img)

    # 计算截屏时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"截屏时间: {elapsed_time * 1000} 毫秒")