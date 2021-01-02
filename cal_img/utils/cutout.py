import numpy as np
import cv2

from utils.read_img import get_gray_img
from utils.lbp import origin_LBP
from utils import crop


def cut_out_pic(img_path):
    """
    使用了一个遮罩层，使得转换出来的 LBP 特征图片不会有背景黑条

    :param img_path: 图像路径
    :return: 切出来的 LBP 特征图片
    """
    gray_img = get_gray_img(img_path)
    # cv2.imwrite("D://0-gray_img.jpg", gray_img)

    _, img = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("D://1-binary_img.jpg", img)

    mask = np.array(img == 255).astype(np.int)

    cut_pic = mask * gray_img
    # cv2.imwrite("D://2-cut_pic.jpg", cut_pic)

    return gray_img, cut_pic

    # lbp_img = origin_LBP(cut_pic, lib=True)
    # lbp_img = sobel.sobel_cal(gray_img, open_cv=True) * mask
    # cv2.imwrite("D://3-LBP_img.jpg", lbp_img)

    # cropped_pic = crop.edge_cutting(cut_pic, lbp_img)
    # cv2.imwrite("D://4-cropped_pic.jpg", cropped_pic)

    # padding_x = int(read_ini.config.get_par("Pic-Par", "padding_x"))
    # padding_y = int(read_ini.config.get_par("Pic-Par", "padding_y"))
    # padding_pic = padding.pad_to(cropped_pic, padding_x, padding_y)
    # cv2.imwrite("D://5-padding_pic.jpg", padding_pic)
    #
    # # resized_pic = cv2.resize(padding_pic, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    # # cv2.imwrite("D://6-resized_pic.jpg", resized_pic)
    #
    # crop_center_x = int(read_ini.config.get_par("Pic-Par", "crop_center_x"))
    # crop_center_y = int(read_ini.config.get_par("Pic-Par", "crop_center_y"))
    # result_pic = crop.crop_center(cropped_pic, crop_center_x, crop_center_y)
    # cv2.imwrite("D://6-result_pic.jpg", result_pic)

    # return cropped_pic

    # if split:
    #     crop_center_x = int(read_ini.config.get_par("Pic-Par", "crop_center_x")) // 2
    #     crop_center_y = int(read_ini.config.get_par("Pic-Par", "crop_center_y")) // 2
    #     result_pic = crop.crop_split(padding_pic, crop_center_x, crop_center_y)
    #     return result_pic


# path = "D://data//origin//train_831//B2F//00011.jpg"
#
# cv2.imwrite("D://12-LBP_img.jpg", cut_out_pic(path))
