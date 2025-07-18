import os
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import sys
from pathlib import Path
# 设置项目根目录
FILE = Path(__file__).resolve() # 获取当前文件的绝对路径
ROOT = FILE.parents[0] # 获取当前文件的父目录
if str(ROOT) not in sys.path: # 
    sys.path.append(str(ROOT)) # 将父目录添加到系统路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # 计算相对路径

import cv2
from PIL import Image
import numpy as np
import hutils.infer.utility as utility
from hutils.infer.predict_rec import TextRecognizer
from hutils.infer.predict_det import TextDetector


class HHOCR:
    def __init__(self):
        # 初始化OCR识别引擎
        # 1. 解析命令行参数
        args = utility.parse_args()
        # 2. 创建文本识别器实例
        self.text_recognizer = TextRecognizer(args)
        self.text_detector = TextDetector(args)
        # 3. 模型预热（避免首次推理延迟）
        # 创建随机测试图像：40*320*3
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        # 进行两次批量推理（每次6张图像）
        for i in range(2):
            res = self.text_recognizer([img] * 6)

    def rec(self, img_list):

        im = cv2.imread(r"C:\Users\hhkj\Desktop\20250714\hhocr\Snipaste_2025-07-10_14-05-56.png")
        r, _ = self.text_detector(im)
        '''
        对输入的图像列表进行文本识别
        参数： 
            img_list: 包含多个图像（numpy数组）的列表
        返回：
            rec_results: 包含多个识别结果的列表，每个结果为一个元组（识别的字符，置信度分值）
        '''
        # 1. 处理空输入
        if (len(img_list) == 0):
            return []
        img_list_rev = [] # 存储预处理后的图像
        # 2. 图像预处理
        for i in range(len(img_list)):
            img = img_list[i]
            # 3. 检测并处理竖排文本
            if (img.shape[0] >= img.shape[1] * 1.3):
                # 3.1 转置图像（将高度变为宽度，宽度变为高度）
                img = cv2.transpose(img)
                # 3.2 上下翻转图像（纠正方向）
                img_rev = cv2.flip(img, 0) # 0：垂直翻转，1：水平翻转，-1：水平垂直翻转
                img_list_rev.append(img_rev)
            else:
                # 4. 横排文本直接使用
                img_list_rev.append(img)
        # 5. 批量文本识别
        rec_results, _ = self.text_recognizer(img_list_rev)
        return rec_results











