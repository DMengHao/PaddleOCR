import os
import sys
import cv2
import numpy as np
from PIL import Image
import hhocr
import time


if __name__ == '__main__':
    
    ocr = hhocr.HHOCR()    # 加载模型以及预热模型
    img0 = cv2.imread(r'C:\Users\hhkj\Desktop\20250714\hhocr\images\001.jpg')
    img1 = cv2.imread(r'C:\Users\hhkj\Desktop\20250714\hhocr\images\002.jpg')
    img2 = cv2.imread(r'C:\Users\hhkj\Desktop\20250714\hhocr\images\003.jpg')
    img3 = cv2.imread(r'C:\Users\hhkj\Desktop\20250714\hhocr\images\004.jpg')
    img_list = []
    img_list.append(img0)
    img_list.append(img1)
    img_list.append(img2)
    img_list.append(img3)
    img_list.append(img2)
    img_list.append(img3)
    img_list.append(img3)
    for i in range(30):
        start_time = time.time()
        rec_results = ocr.rec(img_list)  #  模型推理获得结果 rec_results: [(识别的字符(->str)，置信度分值(->float))，(识别的字符，置信度分值)， ...]
        print(f"##########################{(time.time()-start_time)*1000:.2f}ms")
    print(rec_results)
    
    
