from utils import lbp, cutout, crop
from utils.read_img import get_gray_img

from skimage.feature import local_binary_pattern

import cv2

path = "D://data//origin//train_831//B2F//00001.jpg"
gray_img, cut_pic = cutout.cut_out_pic(path)

cut_out_pic = crop.edge_cutting(cut_pic, cut_pic)
# cv2.imwrite("D://d2.jpg", cut_out_pic)

# org_lbp = lbp.origin_LBP(img)
# cv2.imshow('org_lbp', org_lbp)
# cv2.waitKey(0)
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数

org_lbp = local_binary_pattern(cut_out_pic, n_points, radius)

cv2.imwrite("D://d3.jpg", org_lbp)
# cv2.imshow('org_lbp', org_lbp)
# cv2.waitKey(0)
