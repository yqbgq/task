# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_cal_sobel
#   Author      : HuangWei
#   Created date: 2020-12-20 17:23
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from utils.read_img import get_gray_img
from utils.cal_sobel_gradient_his import sobel_cal_gradient_his

import cv2

path = "D://data//origin//train_831//B2F//00001.jpg"
img = get_gray_img(path)

sobel_img = sobel_cal_gradient_his(img)

# cv2.imshow("test", sobel_img)
# cv2.waitKey(0)
