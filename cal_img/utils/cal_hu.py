# ================================================================
#
#   Editor      : Pycharm
#   File name   : cal_hu
#   Author      : HuangWei
#   Created date: 2020-12-24 20:29
#   Email       : 446296992@qq.com
#   Description : 计算图片的 HU 矩
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import numpy as np
import cv2


def get_img_hu(img: np.ndarray):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments)

    hu = np.array((hu - np.mean(hu)) / np.std(hu))

    return np.abs(hu.reshape(7))
