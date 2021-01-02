"""
@Time : 2020-11-26
@Update : 2020-11-27
@Author : 黄伟
"""

import numpy as np
import cv2


def conv2D(img: np.ndarray, model: np.ndarray, padding: int = 1, step: int = 1,
           gray_limit=True, binary=False, open_cv=False) -> np.ndarray:
    """
    一个很简单的卷积函数

    :param open_cv: 是否使用 open_cv 版本，会更快一点
    :param binary: 是否进行二值化
    :param gray_limit: 限制灰度级在 0-255
    :param step: 卷积核移动的步长
    :param padding: 图片填充的大小，使用 0 进行填充
    :param img: 输入的原图片
    :param model: 卷积模版
    :return: 返回卷积所得，应当是滤波后的结果，即纹理图片
    """
    if open_cv:
        out = cv2.filter2D(img, -1, model)
        if binary:
            ret, out = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        return out

    # ================ 图片、卷积核的基本信息：长、宽============================
    img_width = img.shape[1]
    img_height = img.shape[0]
    model_width = model.shape[1]
    model_height = model.shape[0]
    # ================ 图片、卷积核的基本信息：长、宽============================

    # ================ 计算卷积结果的长、宽，并初始化卷积结果======================
    out_width = (img_width + 2 * padding - model_width) // step + 1
    out_height = (img_height + 2 * padding - model_height) // step + 1
    out = np.zeros([out_height, out_width])
    # ================ 计算卷积结果的长、宽，并初始化卷积结果======================

    for x in range(padding):  # 进行图像的填充
        img = np.row_stack([img, np.zeros(img_width + x)])  # 图片的下方添加一行全0
        img = np.row_stack([np.zeros(img_width + x), img])  # 图片的上方添加一行全0
        img = np.column_stack([img, np.zeros(img_height + 2 * (x + 1))])  # 图片的右侧添加一列全0
        img = np.column_stack([np.zeros(img_height + x + 2 * (x + 1)), img])  # 图片的左侧添加一列全0

    for i in range(out_height):  # 进行卷积计算，这该怎么进行加速啊，如果直接调库算不算作弊............
        for j in range(out_width):
            # 获取以当前像素为左上角的，和卷积核大小相同的块
            temp = img[step * i:step * i + model_height, step * j:step * j + model_width] * model

            # 如果进行灰度限制的话，则将其在上限 255 时进行截断
            if gray_limit:
                out[i, j] = min(np.sum(temp), np.asarray(255))
            else:
                out[i, j] = np.sum(temp)

            # 如果进行二值化的话，没有达到255的部分，全部设置为0，主要是方便拉普拉斯进行边缘检测时变得更加明显，且计算方便
            # 但好像会变慢啊，惆怅
            if binary and out[i, j] != 255:
                out[i, j] = 0

    return out

# a = np.random.random([4, 4])
# b = np.ones([2, 2])
# c = conv(a, b, 1, 2)
# print(a)
# print(c)
