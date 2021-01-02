from sklearn import svm

import numpy as np

from utils import read_ini
from utils.print_info import print_info


# def get_opt():
#     histogram = read_ini.config.get_par("Train-Par", "histogram")
#
#     glcm = read_ini.config.get_par("Train-Par", "glcm")
#
#     simple_color = read_ini.config.get_par("Train-Par", "simple_color")
#
#     histogram_opt = (histogram == "True")
#
#     glcm_opt = (glcm == "True")
#
#     simple_color_opt = (simple_color == "True")
#
#     return histogram_opt, glcm_opt, simple_color_opt


def train(data, label, pars=None):
    # histogram_opt, origin_opt, glcm_option, combine_opt, split_opt = get_opt()
    if pars is None:
        degree = int(read_ini.config.get_par("Train-Par", "degree"))
        coef0 = int(read_ini.config.get_par("Train-Par", "coef0"))
        c = int(read_ini.config.get_par("Train-Par", "C"))
        div = int(read_ini.config.get_par("Train-Par", "div"))
    else:
        degree, coef0, c, div = pars

    # print_info("开始加载训练数据集")
    #
    # data, label = datasets.load_data(train=True, origin=origin_opt, histogram_opt=histogram_opt, gclm=glcm_option,
    #                                  combine=combine_opt, split=split_opt)
    clf = svm.SVC(kernel="poly", degree=degree, coef0=coef0, C=c / div, probability=True)

    clf.fit(data, label)

    return clf


def test(data, label, clf_model):
    # histogram_opts, glcm_opts, simple_color_opts = get_opt()
    # data, labels = datasets.load_data(train=False, origin=origin_opt, histogram_opt=histogram_opt, gclm=glcm_option,
    #                                   combine=combine_opt, split=split_opt)

    result = clf_model.predict(data)
    result = np.array(result)
    acc = np.mean((result == label).astype(np.int))

    return acc

# if __name__ == "__main__":
#     datasets = dataset()
#     clf_model = train()
#     acc_rate = test()
#     print(acc_rate)
