# ================================================================
#
#   Editor      : Pycharm
#   File name   : txt_utils
#   Author      : HuangWei
#   Created date: 2020-12-14 14:26
#   Email       : 446296992@qq.com
#   Description : 从转换后的 TXT 文本中读取数据
#
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import os
import numpy as np

from utils import read_ini

data_root = read_ini.config.get_par("Img-Par", "data_root")
train_path = os.path.join(data_root, "train_")
test_path = os.path.join(data_root, "test_")


def get_data_from_text(path):
    data = []
    with open(path, "r") as f:
        line = f.readline()
        while line:
            item = line
            if "\n" in line:
                item = line.replace("\n", "")

            item = item.split(", ")
            item = [float(x) for x in item]

            data.append(item)

            line = f.readline()
    return data


def get_data(feature, training=True):
    if training:
        path = train_path
    else:
        path = test_path

    dataset, label = [], []

    if type(feature) == str:
        label = np.array(get_data_from_text(path + "label.txt")).reshape(-1)
        dataset = get_data_from_text(path + feature + ".txt")
        return dataset, label
    elif type(feature) == list:
        label = np.array(get_data_from_text(path + "label.txt")).reshape(-1)
        for i in range(len(feature)):
            if i == 0:
                dataset = np.array(get_data_from_text(path + feature[i] + ".txt"))
            else:
                dataset = np.hstack([dataset, get_data_from_text(path + feature[i] + ".txt")])
        return dataset, label


# a = get_data("his_data", training=True)
# print("OK!")