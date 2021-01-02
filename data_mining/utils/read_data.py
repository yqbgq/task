"""
@Time 2020-12-5
@Author 黄伟
@Describe 读取数据，从 data.csv 或者 issues.csv 中读取。前者是癫痫病人数据集，后者是微软操作序列数据集（似乎）
"""

import csv
from typing import List, Tuple
import time
from . import log_tool
from data_mining.utils.word_vector import vector


def get_data(path="./data/data.csv", binary=False, repeat_opt=True) -> Tuple:
    """
    读取 CSV 数据集
    :param repeat_opt: 是否要重复进行平衡数量
    :param path: 数据集存放位置
    :param binary: 是否将标签进行二值化
    :return: 返回从 CSV 中已经读取了的数据
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = []  # type: List[List]
        labels = []  # type: List[int]
        count = 0

        start = time.time()
        log_tool.print_info("Start to load the dataset!")
        for row in reader:
            if count == 0:  # 过滤掉第一行的表头
                count += 1
            else:
                points = [int(x) for x in row[1:-1]]  # 数据点
                label = int(row[-1])  # 标签，1 表示癫痫发作，其余2、3、4和5表示正常人的行为
                if binary and label != 1:  # 是否要进行二值化，可以二值化为 0 表示正常， 1 表示癫痫发作
                    label = 0
                if label == 1 and repeat_opt:
                    repeat = 4
                else:
                    repeat = 1

                for _ in range(repeat):
                    data.append(points)  # 装入数据点
                    labels.append(label)  # 装入标签
        end = time.time()
        log_tool.print_info("Successfully to load the dataset! cost {} second".format((end - start)))
    return data, labels


def get_issues_data(path="./data/issues.csv"):
    """
    读取 issues.csv 数据集

    :param path: 文件路径
    :return: 返回从 CSV 中已经读取了的数据
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = []  # type: List[List]
        labels = []  # type: List[int]
        records = []
        count = 0

        start = time.time()
        log_tool.print_info("Start to load the dataset!")
        for row in reader:
            if count % 200 == 0:
                log_tool.print_info("Processed {} rows data".format(count))
            if count == 0:  # 过滤掉第一行的表头
                count += 1
            else:
                sentence = row[0]                                       # 读取语句
                sentence_vector = vector.get_sentence_vector(sentence)  # 获取语句的向量
                backup = row[2]                                         # 获取该文件对应的备份策略

                # 将备份策略转换成标签
                if backup in records:
                    labels.append(records.index(backup))  # 装入标签
                else:
                    labels.append(len(records))  # 装入标签
                    records.append(backup)

                data.append(sentence_vector)  # 装入数据点

                count += 1
        end = time.time()
        log_tool.print_info("Successfully to load the dataset! cost {} second".format((end - start)))
    return data, labels
