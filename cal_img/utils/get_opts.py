# ================================================================
#
#   Editor      : Pycharm
#   File name   : get_opts
#   Author      : HuangWei
#   Created date: 2020-12-11 11:23
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from utils import read_ini


def get_train_opts():
    histogram = read_ini.config.get_par("Train-Par", "histogram")

    # glcm = read_ini.config.get_par("Train-Par", "glcm")

    simple_color = read_ini.config.get_par("Train-Par", "simple_color")

    # hsv_color = read_ini.config.get_par("Train-Par", "hsv_color")

    # hog = read_ini.config.get_par("Train-Par", "hog")

    sobel = read_ini.config.get_par("Train-Par", "sobel")

    # hu = read_ini.config.get_par("Train-Par", "hu")

    sift = read_ini.config.get_par("Train-Par", "sift")

    histogram_opt = (histogram == "True")

    # glcm_opt = (glcm == "True")

    simple_color_opt = (simple_color == "True")

    # hsv_opt = (hsv_color == "True")

    # hog_opt = (hog == "True")

    sobel_opt = (sobel == "True")

    # hu_opt = (hu == "True")

    sift_opt = (sift == "True")

    return histogram_opt, simple_color_opt, sift_opt, sobel_opt
