import copy

import numpy as np
import math

from method import multi_class_svm
from utils.load_data import get_data
from script import preprocess_img
from utils.print_info import print_info

features = ["his_data", "color_data", "sobel_data"]
boost_pars = [3, 22, 48, 1]
pars = [3, 22, 35, 1]


def get_weight(clf, al_train_data, al_train_label):
    acc = multi_class_svm.test(al_train_data, al_train_label, clf)
    weight = 0.5 * math.log(acc / (1 - acc), 10)
    return weight


def get_total_acc(clf_list, data, label):
    result = np.ndarray([0])
    for i in range(len(clf_list)):
        if i == 0:
            result = clf_list[i][1].predict_proba(data) * clf_list[i][0]
        else:
            result = result + clf_list[i][1].predict_proba(data) * clf_list[i][0]
    result = result
    result = np.argmax(result, axis=1)
    check = (result == label)
    acc = np.sum(check) / label.shape[0]
    return acc, check


def reallocate(clf_list, al_train_data, al_train_label):
    _, check = get_total_acc(clf_list, al_train_data, al_train_label)

    false_idx = np.argwhere(check == False).reshape(-1)
    false_sample = al_train_data[false_idx]
    false_label = al_train_label[false_idx]
    al_train_data = np.vstack([al_train_data, false_sample])
    al_train_label = np.hstack([al_train_label, false_label])
    return al_train_data, al_train_label


def assemble_learning(par):
    last_acc = 0
    clf_list = []
    al_train_data, al_train_label = copy.copy(train_data), copy.copy(train_label)
    while True:
        clf = multi_class_svm.train(al_train_data, al_train_label, pars=par)
        weight = get_weight(clf, al_train_data, al_train_label)
        clf_list.append([weight, clf])
        total_acc, _ = get_total_acc(clf_list, test_data, test_label)
        if (total_acc <= last_acc or abs(total_acc - last_acc) < 0.05) and len(clf_list) >= 3:
            clf_list.pop()
            return clf_list, last_acc
        else:
            last_acc = total_acc
            al_train_data, al_train_label = reallocate(clf_list, al_train_data, al_train_label)


if __name__ == "__main__":
    print_info("开始进行图片切割，该任务可能需要持续 15 - 20分钟，可以使用已经切割转换好的TXT文件")

    # preprocess_img.preprocess_img()

    print_info("图片切割完成，开始训练")

    print_info("开始加载数据集")
    train_data, train_label = get_data(feature=features, training=True)
    test_data, test_label = get_data(feature=features, training=False)
    print_info("数据集加载完成")

    while True:
        clf_type = input("请选择分类器类型： 1. 单分类器学习 2. 集成学习\n")
        if clf_type == "1" or clf_type == "2":
            break

    if clf_type == "1":
        clf = multi_class_svm.train(train_data, train_label, pars=pars)
        acc = multi_class_svm.test(test_data, test_label, clf)
    else:
        clf_list, acc = assemble_learning(boost_pars)

    print_info("准确率为 {:.2f}".format(acc * 100))
