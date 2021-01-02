"""
@Time : 2020-11-26
@Update : 2020-11-27
@Author : 黄伟
"""

import numpy as np
from typing import List


def histogram_gray_cal_s(images: List[np.ndarray]):
    result_list = []
    for img in images:
        result_list.append(histogram_gray_cal(img))
    result = np.array(result_list)
    return result.flatten()


def histogram_gray_cal(image: np.ndarray, np_opt=True):
    image = image.astype(np.int)
    height = image.shape[0]
    width = image.shape[1]
    if np_opt:
        image.resize(image.shape[0] * image.shape[1])
        his = np.bincount(image, minlength=256) / (height * width)
        his_list = [(his[2 * i] + his[2 * i + 1]) / 2 for i in range(128)]
        return np.array(his_list)
    result = [0 for _ in range(256)]

    for i in range(height):
        for j in range(width):
            result[image[i, j]] += 1
    return np.array(result) / (height * width)


def histogram_cal(image: np.ndarray):
    """
    输入一张灰度图片，计算在 X 方向和 Y 方向上的灰度值之和的直方图
    昨天写的时候脑子抽了用循环，改进性能的时候，发现用我不是求灰度直方图啊，我晕了
    直接用 numpy 的求和不是很舒服吗

    :param image: 灰度图片
    :return: X 方向上的直方图和 Y 方向上的灰度值之和的直方图
    """

    data_x = np.sum(image, axis=0)  # type: np.ndarray
    data_y = np.sum(image, axis=1).T  # type: np.ndarray

    return data_x, data_y
