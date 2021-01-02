"""
@Time 2020-12-3\n
@Describe DBSCAN 算法，对算法的聚类结果进行了可视化，由于算法的原因，我认为会存在一些不是核心点的样本没有被聚类进簇中可以有相应的处理方法，但在这里就不作更多的补充了\n
@Author 黄伟
"""

import numpy as np
from typing import Dict, List
from collections import Counter

from data_mining.utils import read_data
from data_mining.utils import cal_dis


class dbscan:
    def __init__(self, min_points, eps, issues=True):
        """
        初始化函数

        :param min_points: 距离 eps 内最少需要 min_points 个样本才能成为核心点
        :param eps: 距离大小
        """
        self.min_points = min_points
        self.eps = eps
        self.issues = issues

        # 簇字典，存储样本点
        self.clusters = {}  # type: Dict[int, List]
        # 簇字典，存储样本索引
        self.cluster_indexes = {}  # type: Dict[int, List]
        # 噪声点集合
        self.noise = []

    def fit(self):
        """进行拟合"""
        self.__fit()  # 进行拟合
        self.__judge()  # 展示拟合结果

    def __fit(self):
        self.__get_data()  # 获取相应的数据
        self.dis_array = cal_dis.cal_dis(self.data, use_matrix=True)  # 计算距离矩阵，即每个样本到其他样本之间的距离
        self.__check_point()  # 检查每一个样本点的类型，是噪声、边缘点还是核心点
        self.__remove_noise()  # 去除噪点
        self.__process()  # 处理每一个样本点

    def __get_data(self):
        """
        为程序设置测试数据
        """
        if self.issues:
            data, labels = read_data.get_issues_data()
        else:
            data, labels = read_data.get_data(binary=True, repeat_opt=True)
        self.data = np.asarray(data)  # type: np.ndarray
        self.label = np.asarray(labels)

        # 每个样本的标记数组 0: 未处理 1: 噪点 2: 边界点 3+: 核心点代表的簇号
        self.mark = [0 for _ in range(self.data.shape[0])]
        # 每个样本点所属簇的簇号
        self.cluster_of_point = np.array([-1 for _ in range(self.data.shape[0])])

    def __get_nearest_point(self, i):
        """
        获取索引为 i 的样本点符合距离特征的样本索引

        :param i:   索引编号 i
        :return:    满足条件的索引标号集合
        """
        dis = self.dis_array[i]
        indexes = np.where(dis <= self.eps)[0]
        indexes = indexes[indexes != i]
        return indexes

    def __check_point(self):
        """检查某个样本点的类型"""
        # 1 是噪点 2 是边缘点 3 是核心点
        for i in range(self.data.shape[0]):
            # 符合距离限制的样本索引集合
            dis_less_than_limit = self.__get_nearest_point(i)  # type: np.ndarray

            # 进行标记
            if dis_less_than_limit.shape[0] == 0:
                self.mark[i] = 1
            elif dis_less_than_limit.shape[0] < self.min_points:
                self.mark[i] = 2
            else:
                self.mark[i] = 3

        self.mark = np.array(self.mark)  # 处理为 numpy 的数组
        self.kernel_point_indexes = np.where(self.mark == 3)  # 所有核心点的索引

    def __remove_noise(self):
        """将噪声点加入到噪声列表中"""
        indexes = np.where(self.mark == 1)[0]
        for index in indexes:
            self.noise.append(self.data[index: index + 1])

    def __process(self):
        """对样本进行处理，遍历所有的核心点进行处理"""
        for index in self.kernel_point_indexes[0]:
            # 如若样本点所属的簇为 -1 即还没有归类，则处理该点
            if self.cluster_of_point[index] == -1:
                self.clusters[index] = []  # 新建簇的集合
                self.cluster_indexes[index] = []  # 新建簇的集合
                self.cluster_of_point[index] = index  # 将该核心点归为 index 类
                self.clusters[index].append(self.data[index: index + 1])  # 将该样本点加入到簇中
                self.cluster_indexes[index].append(index)  # 将该样本点加入到簇中
                self.__process_kernel(index, index)  # 处理该核心点

    def __process_kernel(self, start, num_of_cluster):
        """
        处理该核心点

        :param start:   核心点的索引
        :param num_of_cluster: 该核心点所属簇编号
        """
        near_kernel_indexes = []  # 该核心点周围 eps 内的核心点的索引号
        near_point_indexes = self.__get_nearest_point(start)  # 获取该核心点周围符合距离限制的点的索引

        for index in near_point_indexes:
            if self.cluster_of_point[index] == -1:  # 该核心点是否还未有归属
                self.cluster_of_point[index] = num_of_cluster
                self.clusters[num_of_cluster].append(self.data[index: index + 1])
                self.cluster_indexes[num_of_cluster].append(index)
                if self.mark[index] == 3:
                    near_kernel_indexes.append(index)

        for index in near_kernel_indexes:
            self.__process_kernel(index, num_of_cluster)  # 递归处理

    def __judge(self):
        count = 0

        for key in self.clusters.keys():
            cluster = self.clusters[key]
            count += len(cluster)

        print("样本总数：", self.data.shape[0], "簇数量：", len(self.clusters.keys()))
        print("簇中样本数量：", count, "噪声点数量：", len(self.noise))
        print("未在聚类中的样本数量：", self.data.shape[0] - count - len(self.noise))

        num = 0
        right_sample = 0
        for key in self.clusters.keys():
            cluster_labels = self.label[self.cluster_indexes[key]]
            counter = Counter(cluster_labels)
            most_counter = counter.most_common(1)[0][0]
            result = np.asarray(cluster_labels == most_counter).astype(np.int32)
            info = "第 {} 个簇中的主要元素为 {} 类，其中".format(num, most_counter)
            for x in counter:
                info += "第 {} 类有 {} 个样本，".format(x, counter[x])
            info += "簇分类的准确率为{:.2f}%".format((np.sum(result) / result.shape[0]) * 100)
            print(info)
            num += 1
            right_sample += np.sum(result)
        print("总体分类正确率为 {:.2f}%\n".format((right_sample / count) * 100))

# dd = dbscan(50, 2)
# dd.fit()
# print("ok")
