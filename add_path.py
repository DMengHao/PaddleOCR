# 读取原始文件
with open('D:/PaddleOCR-2.6-release-2.6/train_data/rec_20241111/rec_gt_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 在每行前添加"C:/"
modified_lines = ['D:/PaddleOCR-2.6-release-2.6/train_data/rec_20241111/' + line for line in lines]

# 写入新文件
with open('D:/PaddleOCR-2.6-release-2.6/train_data/rec_20241111/rec_gt_test_output.txt', 'w', encoding='utf-8') as f:
    f.writelines(modified_lines)