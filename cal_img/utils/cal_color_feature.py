# ================================================================
#   Copyright, All Rights Reserved
#
#   Editor      : Pycharm
#   File name   : cal_color_feature
#   Author      : HuangWei
#   Created date: 2020-12-10 17:44
#   Email       : 446296992@qq.com
#   Description : 计算颜色直方图
#
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import time

import cv2
import numpy as np
from utils import read_ini


def read_img(img_path):
    scale_x = float(read_ini.config.get_par("Pic-Par", "color_x"))
    scale_y = float(read_ini.config.get_par("Pic-Par", "color_y"))

    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    return img


def cal_hsv(img_path):
    img = read_img(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = np.resize(img_hsv[:, :, 0], img.shape[1] * img.shape[0])
    s = np.resize(img_hsv[:, :, 1], img.shape[1] * img.shape[0])
    v = np.resize(img_hsv[:, :, 2], img.shape[1] * img.shape[0])

    return h, s, v


def cal_hsv_color_hist(img_path):
    h, s, v = cal_hsv(img_path)

    h_hist, s_hist, v_hist = cal_hist(h, s, v)

    return np.hstack([h_hist, s_hist, v_hist])


def cal_rgb(img_path):
    """
    读取图片的 RGB 通道信息
    :param img_path: 图片路径
    :return: RGB通道信息
    """
    img = read_img(img_path)

    width = img.shape[1]
    height = img.shape[0]

    r = np.resize(img[:, :, 0], width * height)
    g = np.resize(img[:, :, 1], width * height)
    b = np.resize(img[:, :, 2], width * height)

    return r, g, b


def cal_hist(first, second, third):
    """
    使用三通道信息分别计算三个通道的直方图
    :param first: 第一个通道，可以是 R 也可以是 C1
    :param second: 第二个通道
    :param third: 第三个通道
    :return: 三个通道对应的直方图
    """
    temp_first_hist = np.bincount(first, minlength=256)[8:58]
    temp_second_hist = np.bincount(second, minlength=256)[8:58]
    temp_third_hist = np.bincount(third, minlength=256)[8:58]

    first_hist = [(temp_first_hist[2 * i] + temp_first_hist[2 * i + 1]) / 2 for i in range(25)]
    second_hist = [(temp_second_hist[2 * i] + temp_second_hist[2 * i + 1]) / 2 for i in range(25)]
    third_hist = [(temp_third_hist[2 * i] + temp_third_hist[2 * i + 1]) / 2 for i in range(25)]

    first_hist = np.array(first_hist)
    second_hist = np.array(second_hist)
    third_hist = np.array(third_hist)

    first_hist = first_hist / np.sum(first_hist)
    second_hist = second_hist / np.sum(second_hist)
    third_hist = third_hist / np.sum(third_hist)

    return first_hist, second_hist, third_hist


def cal_simple_color_hist(img_path):
    """
    使用 RGB 通道，计算简单的颜色直方图
    :param img_path: 图片路径
    :return: 叠加变为一维向量的简单颜色直方图
    """
    r, g, b = cal_rgb(img_path)

    result = np.hstack(cal_hist(r, g, b))

    return result


def cal_three_c_color_hist(img_path):
    """
    使用转换后的图片通道信息，转化为颜色直方图
    效果没有简单的颜色直方图好，我悟了，simple is good
    :param img_path: 图片路径
    :return: 叠加变为一维向量的简单颜色直方图
    """
    r, g, b = cal_rgb(img_path)

    c_1 = (r + g + b) // 3
    c_2 = (r + (255 - b)) // 2
    c_3 = (r + 2 * (255 - g) + b) // 4

    c_1_hist, c_2_hist, c_3_hist = cal_hist(c_1, c_2, c_3)

    return np.hstack([c_1_hist, c_2_hist, c_3_hist])


# path = "D://data//origin//train_831//B2F//00001.jpg"
# start = time.time()
# a = cal_hsv_color_hist(path)
# print(time.time() - start)
