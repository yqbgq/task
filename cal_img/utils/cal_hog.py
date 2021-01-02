# ================================================================
#
#   Editor      : Pycharm
#   File name   : cal_hog
#   Author      : HuangWei
#   Created date: 2020-12-17 21:59
#   Email       : 446296992@qq.com
#   Description : 计算图片的 HOG 特征
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from skimage import feature as ft
import numpy as np


def cal_hog(img: np.ndarray):
    width = img.shape[1]
    height = img.shape[0]
    feature = ft.hog(img, pixels_per_cell=[height // 4, width // 4])

    return feature
