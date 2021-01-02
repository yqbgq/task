import numpy as np

from typing import Dict, List
from collections import Counter

from data_mining.utils import read_data
from data_mining.utils import cal_dis


class hierarchical:
    def __init__(self, limit_class_num, issues):
        self.limit_class_num = limit_class_num
        self.issues = issues

        # 簇字典，存储样本点
        self.clusters = {}  # type: Dict[int, List[np.ndarray]]
        # 簇字典，存储样本索引
        self.cluster_indexes = {}  # type: Dict[int, List[int]]

    def fit(self):
        self.__fit()

    def __fit(self):
        self.__get_data()
        self.__init_dict()
        self.dis_array = cal_dis.cal_dis(self.data, use_matrix=True)
        self.__sort_distance()
        self.__process()
        print("OK")
        self.__judge()

    def __init_dict(self):
        n = self.data.shape[0]
        self.clusters = {i: [self.data[i]] for i in range(n)}
        self.cluster_indexes = {i: [i] for i in range(n)}

    def __get_data(self):
        """
        为程序设置测试数据
        """
        if self.issues:
            data, labels = read_data.get_issues_data()
        else:
            data, labels = read_data.get_data(binary=True, repeat_opt=False)

        self.data = np.asarray(data)  # type: np.ndarray
        self.label = np.asarray(labels)

        # 每个样本的标记数组 0: 未处理 1: 噪点 2: 边界点 3+: 核心点代表的簇号
        self.mark = [0 for _ in range(self.data.shape[0])]
        # 每个样本点所属簇的簇号
        self.cluster_of_point = np.array([-1 for _ in range(self.data.shape[0])])  # type: np.ndarray

    def __sort_distance(self):
        n = self.data.shape[0]
        mask = np.ones([n, n])
        mask = (1 - np.triu(mask, 1)).astype(np.bool)
        dis = self.dis_array
        dis[mask] = -1

        dis.resize([n ** 2, 1])
        index = dis.T.argsort()[0]
        self.indexes = index[(1 + n) * n // 2:]

    def __process(self):
        """
        进行聚类，遍历所有最近的点对集合，点对的索引可以通过从一维坐标中进行还原得到
        通过判断簇字典中键的个数，既可以判断当前还剩下多少个聚类
        在进行聚类时，考虑以下四种情况：
            1. 样本 i 和 j 都还没有被包含在某个簇中
            2. 样本 i 没有被包含在某个簇中，但样本 j 已经在某个簇中
            3. 样本 i 已经在某个簇中，但样本 i 没有被包含在某个簇中
            4. 样本 i 和 j 都已经包含在某个簇中了
        """
        n = self.data.shape[0]
        out_list = []
        for index in self.indexes:
            maintain_num = len(self.clusters.keys())
            if maintain_num <= self.limit_class_num:
                break
            if maintain_num % 50 == 0 and maintain_num not in out_list:
                print("现今还存在{}个聚类".format(maintain_num))
                out_list.append(maintain_num)
            i = index // n
            j = index % n
            if self.cluster_of_point[i] == -1 and self.cluster_of_point[j] == -1:
                self.clusters[i].append(self.data[j])
                self.cluster_indexes[i].append(j)
                self.clusters.pop(j)
                self.cluster_indexes.pop(j)
                self.cluster_of_point[i] = i
                self.cluster_of_point[j] = i
            elif self.cluster_of_point[i] == -1 and self.cluster_of_point[j] != -1:
                j_class = self.cluster_of_point[j]
                self.clusters[j_class].append(self.data[i])
                self.cluster_of_point[i] = j_class
                self.cluster_indexes[j_class].append(i)
                self.clusters.pop(i)
                self.cluster_indexes.pop(i)
            elif self.cluster_of_point[i] != -1 and self.cluster_of_point[j] == -1:
                i_class = self.cluster_of_point[i]
                self.clusters[i_class].append(self.data[j])
                self.cluster_of_point[j] = self.cluster_of_point[i]
                self.cluster_indexes[i_class].append(j)
                self.clusters.pop(j)
                self.cluster_indexes.pop(j)
            elif self.cluster_of_point[i] != -1 and self.cluster_of_point[j] != -1:
                i_class = self.cluster_of_point[i]
                j_class = self.cluster_of_point[j]
                if i_class == j_class:
                    continue
                self.clusters[i_class].extend(self.clusters[j_class])
                self.clusters.pop(j_class)
                self.cluster_indexes[i_class].extend(self.cluster_indexes[j_class])
                for j_index in self.cluster_indexes[j_class]:
                    self.cluster_of_point[j_index] = i_class
                self.cluster_indexes.pop(j_class)

    def __judge(self):
        count = 0

        for key in self.clusters.keys():
            cluster = self.clusters[key]
            count += len(cluster)

        print("样本总数：", self.data.shape[0], "簇数量：", len(self.clusters.keys()))

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

# a = hierarchical(32, issues=True)
# a.fit()
