#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 学信图片切割

import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np

import ocr_training
from splitter import Splitter
# 行分割
def segment_and_pred(source_path, print_path):
    print('Start process pic:' + source_path)
    splitter = Splitter()
    image_color = cv2.imread(source_path)
    image_color = cv2.resize(image_color, (image_color.shape[1]*3, image_color.shape[0]*3))

    image_color = cv2.GaussianBlur(image_color, (11, 11), 19)

    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    ret, adaptive_threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    """cv2.threshold()函数的作用是将一幅灰度图二值化,ret是true或false，adaptive_threshold是目标图像"""
    ret, at = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV) #at shape(291, 657)
    cv2.imwrite('black.png', adaptive_threshold)
    # 计算换行内容索引
    """这一步是将有文字的行数提取出来"""

############
    horizontal_sum = np.sum(at, axis=1)#返回矩阵中每一行的元f素相加
    peek_ranges = splitter.extract_peek_ranges_from_array(horizontal_sum)

    line_empty_count = 0
    result = ""
    for i in range(len(peek_ranges)):
        tmp1 = adaptive_threshold[peek_ranges[i - line_empty_count][0]: peek_ranges[i - line_empty_count][1], :]
        splitter.show_img('first image', tmp1)
        """将上面的到的tmp1图像保存为kv0，并存入resources中的degree里面"""
        kv0_path = print_path + str(i) + '/'
        if not os.path.exists(kv0_path):
            os.makedirs(kv0_path)
        cv2.imwrite(kv0_path + 'kv0.png', tmp1)
        space_number=splitter.process_by_path(kv0_path + 'kv0.png', kv0_path, minimun_range=10)

        os.remove(kv0_path + 'kv0.png')
        files = os.listdir(kv0_path)
        num_png = len(files)
        """舍弃一些图片小于1的文件夹"""
        if num_png > 1:
            pred_result, pred_val_list = ocr_training.pred(kv0_path,space_number=space_number)
            if i<len(peek_ranges)-1:
                result=result+pred_result+"\n"
            elif i==len(peek_ranges)-1:
                result=result+pred_result
    return result
# 这段代码是一个名为segment_and_pred的函数，它接受两个参数：source_path和print_path。函数的主要功能是对输入的图片进行分割和文字识别。
#
# 首先，使用OpenCV库读取图片，并将其大小放大3倍。
# 对图片进行高斯模糊处理，以减少噪声。
# 将图片转换为灰度图像，并使用阈值化方法将其二值化。
# 计算每一行的像素和，以确定换行内容的范围。
# 遍历每个范围，提取对应的图像片段，并保存为kv0.png。
# 对每个图像片段进行处理，得到空格数量。
# 删除kv0.png文件。
# 统计文件夹中的图片数量，如果大于1，则进行文字识别。
# 将识别结果拼接成一个字符串，并在每行之间添加换行符。
# 返回最终的识别结果字符串。
# 校验图片size
def check_img(path, img_type):
    i = cv2.imread(path)
    if 'school' in img_type:
        if i.shape[0] >= 378 and (i.shape[0] - 378) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    else:
        if i.shape[0] >= 294 and (i.shape[0] - 294) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    return result
# 这段代码是一个名为check_img的函数，它接受两个参数：path和img_type。函数的主要功能是检查给定路径的图片是否符合特定的尺寸要求。
#
# 首先，使用cv2.imread()函数读取指定路径的图片，并将其存储在变量i中。
# 然后，根据img_type的值进行判断。
# 如果img_type中包含字符串"school"，则检查图片的高度是否大于等于378，高度与378的差值是否是20的倍数，以及图片的宽度是否等于660。
# 如果满足这些条件，将结果存储在一个字典中，其中code为0，desc为空字符串。
# 否则，将结果存储在一个字典中，其中code为-1，desc为"图片size非法"，并打印出该信息。
#
# 如果img_type中不包含字符串"school"，则执行类似的操作，但是检查的条件不同。
# 检查图片的高度是否大于等于294，高度与294的差值是否是20的倍数，以及图片的宽度是否等于660。
#
# 最后，返回结果字典。
if __name__ == '__main__':
    result=segment_and_pred(
        './frame.jpg',
        './resources/test/')