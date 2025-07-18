import argparse
import os
import sys
import platform
import cv2
import numpy as np
import paddle
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference
import time
from ppocr.utils.logging import get_logger



def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = current_dir + '/../../'
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="是否使用GPU加速")
    parser.add_argument("--use_xpu", type=str2bool, default=False, help="是否使用XPU加速（百度昆仑芯片）")
    parser.add_argument("--ir_optim", type=str2bool, default=True, help="是否启动IR优化（中间表示优化）")
    parser.add_argument("--use_tensorrt", type=str2bool, default=False, help="是否使用TensorRT加速推理")
    parser.add_argument("--min_subgraph_size", type=int, default=15, help="TensorRT子图最小节点数")
    parser.add_argument("--precision", type=str, default="fp32", help="推理精度，支持fp32、fp16、int8")
    parser.add_argument("--gpu_mem", type=int, default=500, help="GPU显存分配大小MB")

    # params for text detector 文本检测参数（通用）
    parser.add_argument("--image_dir", type=str, default=r"C:\Users\hhkj\Desktop\20250714\hhocr\Snipaste_2025-07-10_14-05-56.png", help="输入图像目录路径")
    parser.add_argument("--det_algorithm", type=str, default='DB', help="检测算法（DB/EAST/SAST/PSE/FCE）")
    parser.add_argument("--det_model_dir", type=str, default=r"C:\Users\hhkj\Desktop\20250714\hhocr\model\det", help="检测模型目录路径")
    parser.add_argument("--det_limit_side_len", type=float, default=960, help="图像缩放限制边长")
    parser.add_argument("--det_limit_type", type=str, default='max', help="缩放类型（max/min）")

    # DB parmas DB检测算法参数
    parser.add_argument("--det_db_thresh", type=float, default=0.3, help="二值化阈值")
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6, help="检测框阈值")
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5, help="文本框扩展比例")
    parser.add_argument("--max_batch_size", type=int, default=10, help="最大批处理大小")
    parser.add_argument("--use_dilation", type=str2bool, default=False, help="是否使用膨胀操作")
    parser.add_argument("--det_db_score_mode", type=str, default="fast", help="得分计算模型（fast/slow）")
    # EAST parmas EAST检测算法参数
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8, help="得分阈值")
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1, help="覆盖阈值")
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2, help="NMS阈值")

    # SAST parmas SATA检测算法参数
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5, help="得分阈值")
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2, help="NMS阈值")
    parser.add_argument("--det_sast_polygon", type=str2bool, default=False, help="是否输出多边形框")

    # PSE parmas PSE检测算法参数
    parser.add_argument("--det_pse_thresh", type=float, default=0, help="二值化阈值")
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85, help="检测框阈值")
    parser.add_argument("--det_pse_min_area", type=float, default=16, help="最小检测区域")
    parser.add_argument("--det_pse_box_type", type=str, default='quad', help="检测框类型")
    parser.add_argument("--det_pse_scale", type=int, default=1, help="缩放比例")

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32], help="特征图缩放比例")
    parser.add_argument("--alpha", type=float, default=1.0, help="损失函数权重")
    parser.add_argument("--beta", type=float, default=1.0, help="损失函数权重")
    parser.add_argument("--fourier_degree", type=int, default=5, help="傅里叶级数阶数")
    parser.add_argument("--det_fce_box_type", type=str, default='poly', help="检测框类型")

    # params for text recognizer 文本识别算法
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet', help="识别算法")
    parser.add_argument("--rec_model_dir", type=str, default=root_path+"./model/rec", help="识别模型目录")
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320", help="输入图像形状（通道，高，宽）")
    parser.add_argument("--rec_batch_num", type=int, default=1, help="识别批次大小")
    parser.add_argument("--max_text_length", type=int, default=25, help="最大文本长度")
    parser.add_argument("--rec_char_dict_path", type=str, default=root_path+ "./ppocr/utils/2C_keys.txt", help="字典路径")
    parser.add_argument("--use_space_char", type=str2bool, default=True, help="是否识别空格")
    parser.add_argument("--vis_font_path", type=str, default="./doc/fonts/simfang.ttf", help="可视化字体路径")
    parser.add_argument("--drop_score", type=float, default=0.5, help="识别结果置信度阈值")

    # params for e2e 端到端参数
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet', help="端到端算法")
    parser.add_argument("--e2e_model_dir", type=str, help="端到端模型目录")
    parser.add_argument("--e2e_limit_side_len", type=float, default=768, help="图像限制边长")
    parser.add_argument("--e2e_limit_type", type=str, default='max', help="缩放类型")

    # PGNet parmas PGNet端到端算法参数
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5, help="得分阈值")
    parser.add_argument("--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt", help="字符字典路径")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext', help="验证集路径")
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast', help="运动模式（fast/slow）")

    # params for text classifier 文本分类参数
    parser.add_argument("--use_angle_cls", type=str2bool, default=False, help="是否使用方向分类器")
    parser.add_argument("--cls_model_dir", type=str, help="分类模型目录")
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192", help="分类输入形状")
    parser.add_argument("--label_list", type=list, default=['0', '180'], help="分类标签")
    parser.add_argument("--cls_batch_num", type=int, default=6, help="分类批处理大小")
    parser.add_argument("--cls_thresh", type=float, default=0.9, help="分类阈值")

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False, help="是否启用MKLDNN加速")
    parser.add_argument("--cpu_threads", type=int, default=10, help="CPU线程数")
    parser.add_argument("--use_pdserving", type=str2bool, default=False, help="是否使用Paddle Serving预测")
    parser.add_argument("--warmup", type=str2bool, default=False, help="是否预热模型")

    # SR parmas 超分辨率参数
    parser.add_argument("--sr_model_dir", type=str, help="超分辨率模型目录")
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128", help="输入形状")
    parser.add_argument("--sr_batch_num", type=int, default=1, help="批处理大小")

    # 结果输出参数
    parser.add_argument("--draw_img_save_dir", type=str, default="./inference_results", help="结果图像保存目录")
    parser.add_argument("--save_crop_res", type=str2bool, default=False, help="是否保存裁剪结果")
    parser.add_argument("--crop_res_save_dir", type=str, default="./output", help="裁剪结果保存目录")

    # multi-process 多进程参数
    parser.add_argument("--use_mp", type=str2bool, default=False, help="是否使用多进程")
    parser.add_argument("--total_process_num", type=int, default=1, help="总进程数")
    parser.add_argument("--process_id", type=int, default=0, help="当前进程ID")
    # 性能与日志参数
    parser.add_argument("--benchmark", type=str2bool, default=False, help="是否进行性能评测")   
    parser.add_argument("--save_log_path", type=str, default="./log_output/", help="日志保存路径")

    parser.add_argument("--show_log", type=str2bool, default=True, help="是否显示日志")
    parser.add_argument("--use_onnx", type=str2bool, default=False, help="是否使用ONNX格式模型")
    parser.add_argument("--server_port", type=int, default=1, help="")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'table':
        model_dir = args.table_model_dir
    elif mode == 'ser':
        model_dir = args.ser_model_dir
    elif mode == "sr":
        model_dir = args.sr_model_dir
    elif mode == 'layout':
        model_dir = args.layout_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    if args.use_onnx:
        import onnxruntime as ort
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        sess = ort.InferenceSession(model_file_path)
        return sess, sess.get_inputs()[0], None, None

    else:
        file_names = ['model', 'inference']
        for file_name in file_names:
            model_file_path = '{}/{}.pdmodel'.format(model_dir, file_name)
            params_file_path = '{}/{}.pdiparams'.format(model_dir, file_name)
            if os.path.exists(model_file_path) and os.path.exists(
                    params_file_path):
                break
        if not os.path.exists(model_file_path):
            raise ValueError(
                "not find model.pdmodel or inference.pdmodel in {}".format(
                    model_dir))
        if not os.path.exists(params_file_path):
            raise ValueError(
                "not find model.pdiparams or inference.pdiparams in {}".format(
                    model_dir))

        config = inference.Config(model_file_path, params_file_path)

        if hasattr(args, 'precision'):
            if args.precision == "fp16" and args.use_tensorrt:
                precision = inference.PrecisionType.Half
            elif args.precision == "int8":
                precision = inference.PrecisionType.Int8
            else:
                precision = inference.PrecisionType.Float32
        else:
            precision = inference.PrecisionType.Float32

        if args.use_gpu:
            gpu_id = get_infer_gpuid()
            if gpu_id is None:
                logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            config.enable_use_gpu(args.gpu_mem, 0)
            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=args.max_batch_size,
                    min_subgraph_size=args.
                    min_subgraph_size,  # skip the minmum trt subgraph
                    use_calib_mode=False)

                # collect shape
                trt_shape_f = os.path.join(model_dir,
                                           f"{mode}_trt_dynamic_shape.txt")

                if not os.path.exists(trt_shape_f):
                    config.collect_shape_range_info(trt_shape_f)
                    logger.info(
                        f"collect dynamic shape info into : {trt_shape_f}")
                else:
                    logger.info(
                        f"dynamic shape info file( {trt_shape_f} ) already exists, not need to generate again."
                    )
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f,
                                                               True)
                except Exception as E:
                    logger.info(E)
                    logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")

        elif args.use_xpu:
            config.enable_xpu(10 * 1024 * 1024)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if args.precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if hasattr(args, "cpu_threads"):
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        if mode == 'table':
            config.delete_pass("fc_fuse_pass")  # not supported for table
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        if mode in ['ser', 're']:
            input_tensor = []
            for name in input_names:
                input_tensor.append(predictor.get_input_handle(name))
        else:
            for name in input_names:
                input_tensor = predictor.get_input_handle(name)
        output_tensors = get_output_tensors(args, mode, predictor)
        return predictor, input_tensor, output_tensors, config


def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet"]:
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def check_gpu(use_gpu):
    if use_gpu and not paddle.is_compiled_with_cuda():
        use_gpu = False
    return use_gpu

