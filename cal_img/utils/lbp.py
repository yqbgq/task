import numpy as np
from skimage.feature import local_binary_pattern


# 原始LBP算法：选取中心点周围的8个像素点，大于中心点为1，小于为0，将这些1或0顺时针串成8位二进制，即最终表示
def origin_LBP(img, lib=False):
    if lib:
        radius = 1  # LBP算法中范围半径的取值
        n_points = 8 * radius  # 领域像素点数
        org_lbp = 255 - local_binary_pattern(img, n_points, radius)
        return org_lbp
    dst = np.zeros(img.shape, dtype=img.dtype)
    h, w = img.shape
    start_index = 1
    for i in range(start_index, h - 1):
        for j in range(start_index, w - 1):
            center = img[i][j]
            code = 0
            # 顺时针，左上角开始的8个像素点与中心点比较，大于等于的为1，小于的为0，最后组成8位2进制
            code |= (img[i - 1][j - 1] >= center) << np.uint8(7)
            code |= (img[i - 1][j] >= center) << np.uint8(6)
            code |= (img[i - 1][j + 1] >= center) << np.uint8(5)
            code |= (img[i][j + 1] >= center) << np.uint8(4)
            code |= (img[i + 1][j + 1] >= center) << np.uint8(3)
            code |= (img[i + 1][j] >= center) << np.uint8(2)
            code |= (img[i + 1][j - 1] >= center) << np.uint8(1)
            code |= (img[i][j - 1] >= center) << np.uint8(0)
            dst[i - start_index][j - start_index] = code
    return dst
