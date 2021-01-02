from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2


def get_img_features(img: np.ndarray):  # s为图像路径
    values = []

    img = np.asarray(img, dtype=np.int)

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4] 一共计算了四个方向，你也可以选择一个方向
    # 统计得到glcm
    glcm = greycomatrix(img, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True,
                        normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    # 循环计算表征纹理的参数
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = greycoprops(glcm, prop).flatten()  # type: np.ndarray
        temp = (temp - np.mean(temp)) / np.std(temp)
        values.append(temp)
    result = np.array(values).flatten()

    result = result / 1000
    return result

# path = "D://data//processed//train_831//B2F//00012.jpg"
# a = get_img_features(path)
# print("OK")
