# ================================================================
#
#   Editor      : Pycharm
#   File name   : preprocess_img
#   Author      : HuangWei
#   Created date: 2020-12-11 11:25
#   Email       : 446296992@qq.com
#   Description : 对图像进行预处理
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import sys

sys.path.append('..')

from utils import get_opts
from script import change2cropped
from script import change2color_feature
from script import change2txt


def preprocess_img():
    # """根据训练方法进行预处理"""
    # histogram_opts, glcm_opts, simple_color_opts, combine_opts = get_opts.get_train_opts()
    # if histogram_opts or glcm_opts:
    #     change2cropped.change2cropped_lbp()
    # elif simple_color_opts:
    #     change2color_feature.change2simple_color_hist()
    # elif combine_opts:
    #     change2cropped.change2cropped_lbp()
    #     change2color_feature.change2simple_color_hist()
    change2txt.change()


if __name__ == "__main__":
    preprocess_img()
