# # ================================================================
# #   Copyright, All Rights Reserved
# #
# #   Editor      : Pycharm
# #   File name   : change2color_feature
# #   Author      : HuangWei
# #   Created date: 2020-12-10 23:17
# #   Email       : 446296992@qq.com
# #   Description : 将图片转换为颜色特征
# #
# #    ( ˶˙º˙˶ )୨  Have Fun!!!
# # ================================================================
# import os
# import time
#
# from utils import read_ini
# from utils import cal_color_feature
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
#             saved_path = os.path.join(img_to, name, img_list[i].replace(".jpg", ".txt"))
#             hist_info = cal_color_feature.cal_simple_color_hist(path)
#
#             # hist_info = cal_color_feature.cal_three_c_color_hist(path)
#
#             with open(saved_path, "w") as f:
#                 info = str(list(hist_info))[1:-1]
#                 f.write(info)
#             now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#
#             count += 1
#             if count % 50 == 0:
#                 print(now, "已经完成第", count, "张图片的转换")
#
#
# def change2simple_color_hist():
#     """
#     将图片转化为简单的颜色直方图
#     """
#     test_img_from = read_ini.config.get_par("Img-Par", "origin_test")
#     test_img_to = read_ini.config.get_par("Img-Par", "processed_test")
#     img_from = read_ini.config.get_par("Img-Par", "origin_train")
#     img_to = read_ini.config.get_par("Img-Par", "processed_train")
#
#     change(img_from, img_to)
#     change(test_img_from, test_img_to)
#
#
