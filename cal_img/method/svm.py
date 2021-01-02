from sklearn import svm
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.load_data import dataset


def train():
    epoch = 4
    cls_list = []
    for step in range(epoch):
        print("开始准备第", step, "个分类器的训练")
        samples, labels = datasets.exclude(step, list(range(step)))
        print("加载完成第", step, "个分类器所需数据")
        data = np.asarray(samples)  # type: np.ndarray
        label = np.asarray(labels)
        # cls =
        # cls = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("svm_clf", svm.SVC(kernel="poly", degree=5, coef0=100, C=1.5))
        # ])
        cls = svm.SVC(kernel="linear", C=1)
        print("开始第", step, "个分类器的训练")
        cls.fit(data, label)
        cls_list.append(cls)
        print("完成第", step, "个分类器的训练")
    return cls_list


def test_single(img: np.ndarray, clf_lists: List[svm.SVC]):
    for x in range(4):
        p = clf_lists[x].predict(img)
        if p == 1:
            return x
    return 4


def test(clf_lists):
    data, labels = datasets.load_test_data()
    result = []
    for count in range(data.shape[0]):
        img = data[count]  # type: np.ndarray
        img.resize([1, img.shape[0]])
        predict_class = test_single(img, clf_lists)
        result.append(predict_class)
        print("完成第", count, "张图片的预测，标签为", labels[count], "预测标签为", predict_class)

    result = np.array(result)
    acc = np.mean((result == labels).astype(np.int))
    print(acc)


if __name__ == "__main__":
    datasets = dataset()
    clf_list = train()
    test(clf_list)
