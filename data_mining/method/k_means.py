"""
@Time 2020-12-5
@Author 黄伟
@Describe 使用 k-means 算法对两个数据集进行聚类
"""

from data_mining.utils import read_data
import numpy as np
from collections import Counter


class k_means:
    def __init__(self, class_num=32, issues=True):
        """
        k_means 函数的初始化函数

        :param class_num: 选择 k-means 算法的初始化中心数量
        :param issues:  是否使用 issues 数据集
        """
        self.class_num = class_num
        self.issues = issues

        self.__get_data()   # 加载数据集

        # 在初始化时，选择 class_num 个中心，通过索引获取对应的向量
        index = np.random.choice(self.data.shape[0], self.class_num, replace=False)
        self.class_centers = np.asarray(self.data[index], dtype=np.float)

    def __get_data(self):
        # 判断加载的数据集
        if self.issues:
            data, labels = read_data.get_issues_data()
        else:
            data, labels = read_data.get_data(binary=True, repeat_opt=True)

        # 对数据集进行正则化处理
        self.data = np.asarray(data)
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        self.labels = np.asarray(labels)

    def train(self, step=300):
        """
        进行训练

        :param step: 训练的轮数
        """

        last_acc = 0    # 上一轮的准确率，用于和当前的准确率比较，如果准确率变化太小，则停止迭代

        for x in range(step):
            result = self.__cal_distance()          # 计算各个点和聚类中心之间的距离
            acc = self.__cal_acc(result)            # 计算聚类的准确率

            print(acc)

            if abs(last_acc - acc) <= 0.0001:
                break
            last_acc = acc

            self.__re_center(result)                # 重新计算聚类中心

    def __cal_acc(self, last_result: np.ndarray):
        """
        计算当前聚类的准确率，具体做法是：计算每个簇中同类数量最多的样本，将它的标签作为簇的标签
        计算 簇中该类样本的数量 / 该簇中样本总量

        :param last_result: 最新一次计算的结果
        :return: 准确率
        """

        indexes = []
        right_sample = 0

        for x in range(self.class_num):
            _ = np.argwhere(last_result == x)       # 获取每个聚类中样本的索引
            indexes.append(_.reshape(_.shape[0]))   # 获取相应的索引列表

        # 计算聚类的准确度
        for x in range(self.class_num):
            result_labels = self.labels[indexes[x]]
            counter = Counter(result_labels)
            if len(list(counter.keys())) == 0:
                continue
            print(counter)
            most_counter = counter.most_common(1)[0][0]
            result = np.asarray(result_labels == most_counter).astype(np.int32)
            right_sample += np.sum(result)

        return right_sample / self.labels.shape[0]

    def __cal_distance(self):
        """计算每个点到聚类中心的距离"""

        # 按照公式进行每个样本点到所有聚类中心的计算
        temp_data_0 = np.expand_dims(self.data, len(self.data.shape) - 1).repeat(self.class_num,
                                                                                 len(self.data.shape) - 1)
        temp_data_1 = temp_data_0 - self.class_centers
        temp_data_2 = temp_data_1 ** 2
        temp_data_3 = np.sum(temp_data_2, axis=2)

        # 求最小值所在的索引，即该样本应该归类于哪个聚类中心
        result = np.argmin(temp_data_3, axis=1)
        return result

    def __re_center(self, last_result: np.ndarray):
        """重新计算聚类中心"""
        indexes = []

        # 某个聚类中心的所有样本点的索引集合
        for x in range(self.class_num):
            _ = np.argwhere(last_result == x)
            indexes.append(_.reshape(_.shape[0]))

        # 获取该聚类中心的所有样本点，计算平均值得到新的聚类中心
        for x in range(self.class_num):
            if self.data[indexes[x]].shape[0] == 0:
                continue
            a = np.mean(self.data[indexes[x]], axis=0)
            self.class_centers[x] = a.astype(np.float)


# # 实例化 k-means 类，聚类中心数量为 32， 不使用 issues 数据集
# k = k_means(class_num=32, issues=False)
# # Run！
# k.train()
