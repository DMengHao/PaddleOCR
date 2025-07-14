## 1.标签工具PPOCRLabel环境配置：

```python
conda create -n paddle python==3.9
conda activate paddle
conda install -c conda-forge cudatoolkit=11.8
# 检查cudatoolkit是否安装成功
conda list cudatoolkit
pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# 或者使用镜像源下载
pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall paddleocr 
pip install paddleocr==2.6
pip uninstall numpy
pip install numpy==1.23.5
# 将zlibwapi.dll复制到虚拟环境下的Library/bin和Library/lib下
cp ./zlibwapi.dll ./
# 进入到PaddleOCR目录下 cd ./PaddleOCR-release-2.6/PPOCRLabel 
python PPOCRLabel.py
```

使用教程链接：https://blog.csdn.net/didiaopao/article/details/119652371?fromshare=blogdetail&sharetype=blogdetail&sharerId=119652371&sharerefer=PC&sharesource=&sharefrom=from_link

## 2.训练环境配置

```python
# 进入到./PaddleOCR-release-2.6目录下并进入虚拟环境、
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3.制作数据集

参考链接：https://blog.csdn.net/qq_52852432/article/details/131817619

```python
# 进入到PPOCRLabel文件夹下运行下面命令 生成训练数据
python gen_ocr_train_val_test.py --trainValTestRatio 6:2:2 --dataetRootPath ../train_data/drivingData
```

## 4.训练文字检测模型

```python
# 训练命令 配置文件里面参数自行修改或者根据参考上面的链接，里面有
python tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml
# 测试命令
python tools/infer_det.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model=output/ch_db_driving/best_accuracy.pdparams Global.infer_img="C:\Users\User\Desktop\PaddleOCR-release-2.6\train_data\det\test\"
```

**训练一百轮次的测试结果（网上选取的20张图片自己标注过了）：**



## 5.训练文字识别模型

```python
# 训练命令
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml
# 测试命令
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o Global.pretrained_model=output/rec/best_accuracy.pdparams Global.infer_img=“C:\Users\User\Desktop\PaddleOCR-release-2.6\train_data\rec\test\”
```

**训练一百轮次的测试结果：**

## 6.导出推理模型

```python
# 注：修改对应配置文件的训练模型 和 保存模型的路径
# 文字检测模型
python tools/export_model.py -c "./configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml"
# 文字识别模型
python tools/export_model.py -c "./configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml"
# 推理测试，注意修改路径image_dir 原始图片路径
python tools/infer/predict_system.py --image_dir=":\Users\User\PycharmProjects\PaddleOCR-release-2.6\train_data\drivingData\" --det_model_dir="./inference_model/det/" --rec_model_dir="./inference_model/rec"
```

## 7.简单测试

```python
# 简单测试 修改上面导出的模型和图片路径
from paddleocr import PaddleOCR, draw_ocr
import cv2

ocr = PaddleOCR(use_angle_cls=True, rec_model_dir='D:/PaddleOCR-2.6-release-2.6/output/rec',
                    det_model_dir='D:/PaddleOCR-2.6-release-2.6/output/det')
result = ocr.ocr(cv2.imread("D:/PaddleOCR-2.6-release-2.6/train_data/det/train/Snipaste_2025-07-10_14-06-14.png"), cls=True)
result = result[0]
texts = [print(line) for line in result]
```

