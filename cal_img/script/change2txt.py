# ================================================================
#
#   Editor      : Pycharm
#   File name   : change2txt
#   Author      : HuangWei
#   Created date: 2020-12-14 13:13
#   Email       : 446296992@qq.com
#   Description : 将图片的特征信息转换为 TXT 文本内容
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import os
import time
import multiprocessing

from utils import cal_color_feature, crop, cal_sift, cal_sobel_gradient_his
from utils import get_opts
from utils import histogram
from utils import cutout
from utils import read_ini
from utils.lbp import origin_LBP

test_img_from = read_ini.config.get_par("Img-Par", "origin_test")
img_from = read_ini.config.get_par("Img-Par", "origin_train")
data_root = read_ini.config.get_par("Img-Par", "data_root")

histogram_opt, simple_color_opt, sift_opt, sobel_opt = get_opts.get_train_opts()


class ChangeProcess(multiprocessing.Process):
    def __init__(self, training):
        multiprocessing.Process.__init__(self)
        self.training = training
        self.dataset = {}
        self.labels = []

    def run(self):
        if self.training:
            path = img_from
        else:
            path = test_img_from
        dirs = os.listdir(path)
        count = 0

        if histogram_opt:
            self.dataset["his_data"] = []
        if simple_color_opt:
            self.dataset["color_data"] = []
        if sift_opt:
            self.dataset["sift_data"] = []
        if sobel_opt:
            self.dataset["sobel_data"] = []

        for name in dirs:
            dir_name = os.path.join(path, name)
            img_list = os.listdir(dir_name)

            label = dirs.index(name)

            for i in range(len(img_list)):

                self.labels.append(label)

                img_path = os.path.join(dir_name, img_list[i])
                gray_img, cut_pic = cutout.cut_out_pic(img_path)

                lbp_img = origin_LBP(cut_pic, lib=True)

                cut_out_pic = crop.edge_cutting(cut_pic, lbp_img)

                if simple_color_opt:
                    self.dataset["color_data"].append(cal_color_feature.cal_simple_color_hist(img_path))

                if histogram_opt:
                    self.dataset["his_data"].append(histogram.histogram_gray_cal(cut_out_pic, np_opt=True))

                if sift_opt:
                    self.dataset["sift_data"].append(cal_sift.cal_sift(gray_img))
                if sobel_opt:
                    self.dataset["sobel_data"].append(cal_sobel_gradient_his.sobel_cal_gradient_his(cut_pic))

                count += 1

                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if count % 50 == 0:
                    print(now, "Training: " + str(self.training) + " 已经完成第", count, "张图片的切割转换")
        if self.training:
            for feature in self.dataset.keys():
                write_data(self.dataset, feature, "train")
            write_label("train_label.txt", self.labels)
        else:
            for feature in self.dataset.keys():
                write_data(self.dataset, feature, "test")
            write_label("test_label.txt", self.labels)


def change():

    p1 = ChangeProcess(True)
    p2 = ChangeProcess(False)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # feature_train, label_train = p1.get_result()
    # feature_test, label_test = p2.get_result()
    #
    # print("OK")
    #
    # for feature in feature_train.keys():
    #     write_data(feature_train, feature, "train")
    #
    # for feature in feature_test.keys():
    #     write_data(feature_test, feature, "test")
    #
    # write_label("train_label.txt", label_train)
    #
    # write_label("test_label.txt", label_test)

    print("OK")


def write_data(feature_data, feature, txt_type):
    with open(os.path.join(data_root, "{}_{}.txt".format(txt_type, feature)), "a+") as f:
        for i in range(len(feature_data[feature])):
            info = feature_data[feature][i]
            info = [float(x) for x in info]
            f.write(str(info)[1:-1])
            if i != len(feature_data[feature]) - 1:
                f.write("\n")


def write_label(name, label):
    with open(os.path.join(data_root, name), "a+") as f:
        for i in range(len(label)):
            f.write(str(label[i]))
            if i != len(label) - 1:
                f.write("\n")


if __name__ == "__main__":
    change()
