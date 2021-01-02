# ================================================================
#
#   Editor      : Pycharm
#   File name   : load_color_feature
#   Author      : HuangWei
#   Created date: 2020-12-11 10:01
#   Email       : 446296992@qq.com
#   Description : 从 TXT 文件中读取图片颜色直方图的一维向量
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import numpy as np


def load_data(txt_path):
    with open(txt_path) as f:
        line = f.readline()
        data = [int(x) for x in line.split(",")]
    return np.array(data)
