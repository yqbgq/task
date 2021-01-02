# ================================================================
#
#   Editor      : Pycharm
#   File name   : cal_sift
#   Author      : HuangWei
#   Created date: 2020-12-24 17:03
#   Email       : 446296992@qq.com
#   Description : 计算 SIFT 特征
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import numpy as np
import cv2

from utils import cutout, crop
from utils.lbp import origin_LBP
from utils.read_img import get_gray_img


def cal_sift(img: np.ndarray):
    detector = cv2.xfeatures2d.SIFT_create()
    kps, des = detector.detectAndCompute(img, None)
    gradient_his = np.mean(des, axis=0)
    gradient_his = (gradient_his - np.mean(gradient_his)) / np.std(gradient_his)
    result = [(gradient_his[2 * i] + gradient_his[2 * i + 1]) / 2 for i in range(64)]
    return result

# path = "D://data//origin//train_831//B2F//00112.jpg"
# gray_img, cut_pic = cutout.cut_out_pic(path)
#
# lbp_img = origin_LBP(cut_pic, lib=True)
#
# cut_out_pic = crop.edge_cutting(cut_pic, lbp_img)
# img = get_gray_img(path)
# cal_sift(img)
