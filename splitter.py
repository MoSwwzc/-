#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 文字切割 参考:http://chongdata.com/articles/?p=32
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


class Splitter(object):
    def __init__(self):
        print('Create Splitter')

    # 计算投影分割点
    def extract_peek_ranges_from_array(self, array_vals, minimun_val=5, minimun_range=2):
        """if judge_mini_value!=0:
            minimun_val=judge_mini_value*38"""
        start_i = None
        peek_ranges = []
        for i, val in enumerate(array_vals):
            if val > minimun_val and start_i is None:
                start_i = i
            elif val > minimun_val and start_i is not None:
                pass
            elif val < minimun_val and start_i is not None:
                end_i = i
                if end_i - start_i >= minimun_range:
                    peek_ranges.append((start_i, end_i))
                    start_i = None
                end_i = None
            elif val < minimun_val and start_i is None:
                pass
            else:
                raise ValueError("cannot parse this case...")
        if start_i is not None:
            peek_ranges.append((start_i, i + 1))
        return peek_ranges

    def full_extract(self,image,vertical_peek_ranges):
        # 让白色顶格
        # """这个函数是我加的，目的是为了将字符完整提取出来，没有多余的边界"""
        row_infor=[]
        for i, sub_vertical_peek_ranges in enumerate(vertical_peek_ranges):
            sub_image=image[:,sub_vertical_peek_ranges[0]:sub_vertical_peek_ranges[1]]
            horizontal_sum = np.sum(sub_image, axis=1)
            peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum)
            if len(peek_ranges)>1:
                peek_ranges[0]=(peek_ranges[0][0],peek_ranges[len(peek_ranges)-1][1])
            row_infor.append(peek_ranges[0])
        return row_infor

    def judge_the_space(self,row_infor,vertical_peek_ranges):
        """用来判断相邻两个字符间是否有空格"""
        # 判断有没有空格，字母的高度作为判断，两字母大于七分之三 说明有空格
        if row_infor==[]:
            return []
        space_number=[]
        width = row_infor[0][1] - row_infor[0][0]
        for i in range(len(row_infor)-1):
            distance=vertical_peek_ranges[i+1][0]-vertical_peek_ranges[i][1]
            if distance>width*3/7:
                space_number.append(i)
        if len(space_number)==0:
            return None
        else:
            return space_number

    def fill(self, img, i, result_img_path, sub_segment=False):
        flag = False
        left = 0
        right = 0
        top = 0
        bottom = 0
        expect = 30
        if img.shape[0] < expect and img.shape[0] > 10:
            bottom = int((expect - img.shape[0]) / 2)
            top = expect - bottom - img.shape[0]
            flag = True

        if img.shape[1] < expect and img.shape[1] > 10:
            right = int((expect - img.shape[1]) / 2)
            left = expect - right - img.shape[1]
            flag = True

        # if flag:
        #     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if img.shape[1] < 8 and not sub_segment:
            # 数字字符，切割顶部和底部的空隙
            # 水平投影 水平分割
            horizontal_sum = np.sum(img, axis=1)
            peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum)
            img = img[peek_ranges[0][0]:peek_ranges[len(peek_ranges) - 1][1], 0:img.shape[1]]
            self.show_img('sub img', img)

        # resize resize后效果更好
        # img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_CUBIC)
        if i<10:
            cv2.imwrite(result_img_path +str(0)+ str(i) + '.png', img)
        else:
            cv2.imwrite(result_img_path + str(i) + '.png', img)


    def process_by_img(self, image_color, result_img_path, minimun_range=11, sub_segment=False, pred_val_list=[]):

        # new_shape = (image_color.shape[1] * 2, image_color.shape[0] * 2)
        # image_color = cv2.resize(image_color, new_shape)
        if len(image_color.shape) == 2:
            adaptive_threshold = image_color
        else:
            image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            adaptive_threshold = 255 - image
            """adaptive_threshold = cv2.adaptiveThreshold(image,255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)"""
            #这一步出现问题使得那部分区域加了层边框
            """自适应阈值，类似于二值化。图像的阈值处理一般使得图像的像素值更单一、图像更简单。阈值可以分为全局性质的阈值，
            也可以分为局部性质的阈值，可以是单阈值的也可以是多阈值的。当然阈值越多是越复杂的。"""


        # 水平投影

        vertical_sum = np.sum(adaptive_threshold, axis=0)#将每列加起来
        vertical_peek_ranges = self.extract_peek_ranges_from_array(vertical_sum)

        row_infor=self.full_extract(adaptive_threshold,vertical_peek_ranges)
        i=0
        while i<len(vertical_peek_ranges):
            if vertical_peek_ranges[i][1]-vertical_peek_ranges[i][0]>50 and row_infor[i][1]-row_infor[i][0]>25:
                image=adaptive_threshold[row_infor[i][0]:row_infor[i][1],vertical_peek_ranges[i][0]:vertical_peek_ranges[i][1]]
                horizontal_sum = np.sum(image, axis=1)  # 返回矩阵中每一行的元素相加
                  # 将每行相加为0的舍去
                if vertical_peek_ranges[i][1] - vertical_peek_ranges[i][0] > 170 and not (0 in horizontal_sum):
                    #要找到第二小的值
                    minimun_val= np.sort(horizontal_sum)[1]
                    peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum,minimun_val=minimun_val+1)
                else:
                    peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum)
                for j in range(len(peek_ranges)):
                    if j<len(peek_ranges) and peek_ranges[j][1]-peek_ranges[j][0]<10:
                        del peek_ranges[j]
                image=image[peek_ranges[0][0]:peek_ranges[0][1],:]

                vertical_sum = np.sum(image, axis=0)  # 将每列加起来
                new_vertical_peek_ranges = self.extract_peek_ranges_from_array(vertical_sum)
                new_row_infor = self.full_extract(image, new_vertical_peek_ranges)

                new_vertical_range = []
                new_row_range=[]
                for j in range(len(new_vertical_peek_ranges)):
                    x=vertical_peek_ranges[i][0]+new_vertical_peek_ranges[j][0]
                    y=vertical_peek_ranges[i][0]+new_vertical_peek_ranges[j][1]
                    new_vertical_range.append((x,y))

                    p=row_infor[i][0]+new_row_infor[j][0]
                    q=row_infor[i][0]+new_row_infor[j][1]
                    new_row_range.append((p, q))
                del row_infor[i]
                del vertical_peek_ranges[i]
                for j in range(len(new_vertical_range)):
                    row_infor.insert(j+i, new_row_range[j])
                    vertical_peek_ranges.insert(j+i, new_vertical_range[j])
                t=len(new_vertical_range)
                i=t+i
            else:
                i=i+1
        j=0

        while j < len(vertical_peek_ranges):
            x = vertical_peek_ranges[j][0]
            y = row_infor[j][0]
            w = vertical_peek_ranges[j][1] - x
            h = row_infor[j][1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            if h<10 or w<10 or h>300 or w>300 :
                del vertical_peek_ranges[j]
                del row_infor[j]
                continue
            else:
                sub_img = adaptive_threshold[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                self.fill(sub_img, j, result_img_path, sub_segment=sub_segment)
            j += 1
        space_number = self.judge_the_space(row_infor, vertical_peek_ranges)
        self.show_img('char image', image_color)

        return(space_number)

    def process_by_path(self, source_img_path, result_img_path, minimun_range=11, sub_segment=False, pred_val_list=[]):
        image_color = cv2.imread(source_img_path)
        space_number=self.process_by_img(image_color, result_img_path, minimun_range, sub_segment=sub_segment,
                            pred_val_list=pred_val_list)
        return space_number

    def show_img(self, img_name, img):
        ''
        # cv2.imshow(img_name, img)
        # cv2.waitKey(0)

