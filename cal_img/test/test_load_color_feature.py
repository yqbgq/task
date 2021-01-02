# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_load_color_feature
#   Author      : HuangWei
#   Created date: 2020-12-11 10:04
#   Email       : 446296992@qq.com
#   Description : 测试加载图像颜色直方图函数
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

from utils.load_color_feature import load_data

path = "D://data//processed//train_831//B2F//00012.txt"

data = load_data(path)

print("OK")