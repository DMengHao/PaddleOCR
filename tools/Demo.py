from paddleocr import PaddleOCR, draw_ocr
import cv2

ocr = PaddleOCR(use_angle_cls=True, rec_model_dir='D:/PaddleOCR-2.6-release-2.6/output/rec',
                    det_model_dir='D:/PaddleOCR-2.6-release-2.6/output/det')
result = ocr.ocr(cv2.imread("D:/PaddleOCR-2.6-release-2.6/train_data/det/train/Snipaste_2025-07-10_14-06-14.png"), cls=True)
result = result[0]
texts = [print(line) for line in result]