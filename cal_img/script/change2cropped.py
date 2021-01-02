# """
# @Time : 2020-11-27
# @Author : 黄伟
# """
#
# from utils.read_img import get_gray_img
# from utils import conv
# from utils import crop
# from utils import sobel
# from utils import cutout
# from utils import read_ini
#
# import numpy as np
# import cv2
# import os
# import time
#
#
# def change(img_from, img_to):
#     dirs = os.listdir(img_from)
#     count = 0
#     for name in dirs:
#         dir_name = os.path.join(img_from, name)
#         img_list = os.listdir(dir_name)
#
#         for i in range(len(img_list)):
#             path = os.path.join(dir_name, img_list[i])
#             cut_out_pic = cutout.cut_out_pic(path)
#             saved_path = os.path.join(img_to, name, img_list[i])
#             count += 1
#
#             cv2.imwrite(saved_path, cut_out_pic)
#             now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             if count % 50 == 0:
#                 print(now, "已经完成第", count, "张图片的切割转换")
#             # print("完成第", count, "张图片的转换，耗时", end - start, "秒", path, "======>", saved_path)
#
#
# def change2cropped_lbp():
#     test_img_from = read_ini.config.get_par("Img-Par", "origin_test")
#     test_img_to = read_ini.config.get_par("Img-Par", "processed_test")
#     img_from = read_ini.config.get_par("Img-Par", "origin_train")
#     img_to = read_ini.config.get_par("Img-Par", "processed_train")
#
#     change(img_from, img_to)
#     change(test_img_from, test_img_to)
#
#
# def change2cropped_gray():
#     """为了加速运行训练，需要提前先将原图转换成为切割之后的灰度图"""
#
#     img_from = "D://data//origin//train_831//"
#     img_to = "D://data//processed//train//"
#     # img_from = "D://data//origin//test_351//"
#     # img_to = "D://data//processed//test//"
#
#     dirs = os.listdir(img_from)
#
#     count = 0
#
#     for name in dirs:
#         dir_name = os.path.join(img_from, name)
#         img_list = [os.path.join(dir_name, x) for x in os.listdir(dir_name)]
#
#         for path in img_list:
#             start = time.time()
#             img = get_gray_img(path)
#             sobel_img = sobel.sobel_cal(img, open_cv=True)
#
#             laplace = [
#                 [1, 1, 1],
#                 [1, 8, 1],
#                 [1, 1, 1]
#             ]
#
#             laplace_model = np.asarray(laplace)
#
#             laplace_img = conv.conv2D(img, model=laplace_model, binary=True, open_cv=True)  # type: np.ndarray
#
#             cropped_img = crop.edge_cutting(laplace_img, sobel_img)  # type: np.ndarray
#             cropped_img = cropped_img.astype(np.uint8)
#
#             saved_path = os.path.join(img_to, "{:0>5d}".format(count) + " " + name + ".jpg")
#             count += 1
#
#             cv2.imwrite(saved_path, cropped_img)
#             end = time.time()
#
#             print("完成第", count, "张图片的转换，耗时", end - start, "秒", path, "======>", saved_path)
#
#
# if __name__ == "__main__":
#     # change2cropped_gray()
#     change2cropped_lbp()
