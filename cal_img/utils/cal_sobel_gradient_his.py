# ================================================================
#
#   Editor      : Pycharm
#   File name   : cal_sobel_gradient_his
#   Author      : HuangWei
#   Created date: 2020-12-18 22:11
#   Email       : 446296992@qq.com
#   Description : 激素按索贝尔算子下，梯度直方图
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

import numpy as np
import cv2

from utils import cutout, crop
from utils.read_img import get_gray_img


def sobel_cal_gradient_his(img: np.ndarray, open_cv=True) -> np.ndarray:
    """
    索贝尔算子进行图像灰度检测

    :param open_cv: 是否使用 open_cv 版本的函数，更快无限多倍
    :param img: 灰度图像
    :return: 边缘检测之后的结果
    """

    # 如果使用 open_cv 版本的函数，则调用如下
    if open_cv:
        img = img.astype("uint8")
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(grad_x, 1, grad_y, 1, 0)

        grad.resize([grad.shape[0] * grad.shape[1]])
        his = np.bincount(grad, minlength=256)
        result = np.array([his[2 * i] + his[2 * i + 1] for i in range(10, 128)])
        result = np.array(result) / np.sum(result)
        return result

    width = img.shape[1]
    height = img.shape[0]
    out = np.zeros_like(img)

    # 使用索贝尔算子进行图像处理，一个字，就是慢
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            dx = img[i - 1, j - 1] + 2 * img[i, j - 1] + img[i + 1, j - 1]
            dx -= img[i - 1, j + 1] + 2 * img[i, j + 1] + img[i + 1, j + 1]
            dy = img[i - 1, j - 1] + 2 * img[i - 1, j] + img[i - 1, j + 1]
            dy -= img[i + 1, j - 1] + 2 * img[i + 1, j] + img[i + 1, j + 1]
            out[i, j] = min(255, abs(dx) + abs(dy))

    return out

# path = "D://data//origin//train_831//B2F//00001.jpg"
#
# gray_img, cut_pic = cutout.cut_out_pic(path)
#
# cut_out_pic = crop.edge_cutting(cut_pic, cut_pic)
# sobel_cal_gradient_his(cut_out_pic)