# ================================================================
#   Copyright, All Rights Reserved
#
#   Editor      : Pycharm
#   File name   : test_cal_color_feature
#   Author      : HuangWei
#   Created date: 2020-12-10 22:59
#   Email       : 446296992@qq.com
#   Description : 测试颜色直方图的计算
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

from utils import cal_color_feature
import time

path = "D://data//origin//train_831//B2F//00001.jpg"
start = time.time()
color_his = cal_color_feature.cal_simple_color_hist(path)
print(time.time() - start)  # 0.3580806255340576
print("OK")
