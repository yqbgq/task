"""
@Time : 2020-11-26
@Update : 2020-11-28
@Author : 黄伟
"""

import cv2

from utils import read_ini


def get_gray_img(img_path):
    """
    读取灰度图片

    :param img_path: 图片路径
    :return: 返回 resize 之后的灰度图像结果
    """
    scale_x = float(read_ini.config.get_par("Pic-Par", "scale_x"))
    scale_y = float(read_ini.config.get_par("Pic-Par", "scale_y"))
    color_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(color_img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
    return resized
