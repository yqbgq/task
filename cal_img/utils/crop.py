"""
@Time : 2020-11-26
@Update : 2020-11-27
@Author : 黄伟
"""

import numpy as np
from utils.histogram import histogram_cal


def find_first_and_last(result: np.ndarray):
    """
    通过直方图检测得到的灰度直方图结果，找到列和行的开始和结束，用于切割
    使用了 numpy 之后会快稍微一点点点

    :param result: 灰度直方图结果
    :return: 开始和结束的列号或者行号
    """

    first = (result >= 255 * 20).argmax(axis=0)
    result = result[::-1]  # type: np.ndarray

    last = result.shape[0] - (result >= 255 * 20).argmax(axis=0)
    # for i in range(num):
    #     if result[i] >= 255 * 10:   # 限制必须要多过 10 个边缘点，以防止一些杂乱的噪点
    #         first = i
    #         break
    # for i in range(num):
    #     if result[num - 1 - i] >= 255 * 10:     # 限制必须要多过 10 个边缘点，以防止一些杂乱的噪点
    #         last = num - 1 - i
    #         break
    return first, last


def edge_cutting(laplace_img: np.ndarray, img: np.ndarray):
    """
    传入一个经过拉普拉斯二值化之后的结果，进行边缘切割

    :param laplace_img:
    :param img: 拉普拉斯二值化之后的结果
    :return: 切割之后的结果
    """
    result_x, result_y = histogram_cal(laplace_img)  # 通过拉普拉斯二值化之后的结果，找出进行切割的左右和上下边界
    x_first, x_last = find_first_and_last(result_x)  # X 即列的左右边界
    y_first, y_last = find_first_and_last(result_y)  # Y 即行的上下边界

    cropped_img = img[y_first:y_last + 1, x_first:x_last + 1]  # 进行切割，得到切割后的图片
    return cropped_img  # type: np.ndarray


def crop_center(img: np.ndarray, x=50, y=50):
    width = img.shape[1]
    height = img.shape[0]
    center_img = img[height // 2 - y // 2: height // 2 + y // 2, width // 2 - x // 2: width // 2 + x // 2]
    return center_img


def crop_split(img: np.ndarray, x=50, y=50):
    width = img.shape[1]
    height = img.shape[0]
    center_img = img[height // 2 - y // 2: height // 2 + y // 2, width // 2 - x // 2: width // 2 + x // 2]
    top_left = img[height // 4 - y // 2: height // 4 + y // 2, width // 4 - x // 2: width // 4 + x // 2]
    top_right = img[height // 4 - y // 2: height // 4 + y // 2, width // 4 * 3 - x // 2: width // 4 * 3 + x // 2]
    down_left = img[height // 4 * 3 - y // 2: height // 4 * 3 + y // 2, width // 4 - x // 2: width // 4 + x // 2]
    down_right = img[height // 4 * 3 - y // 2: height // 4 * 3 + y // 2, width // 4 * 3 - x // 2: width // 4 * 3 + x // 2]
    return [center_img, top_left, top_right, down_left, down_right]