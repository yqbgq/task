import csv
import time
from data_mining.utils.word_vector import vector

path = "../data/issues.csv"

# with open(path, 'r') as f:
#     start = time.time()
#     reader = csv.reader(f)
#     sentences = [row[0] for row in reader][1:]
#     backups = [row[2] for row in reader][1:]
#     data = [vector.get_sentence_vector(sentence) for sentence in sentences]
#     labels = []
#
#     records = []
#
#     for backup in backups:
#         # 将备份策略转换成标签
#         if backup in records:
#             labels.append(records.index(backup))  # 装入标签
#         else:
#             labels.append(len(records))  # 装入标签
#             records.append(backup)
#     print("OK", time.time() - start)


# with open(path, 'r') as f:
#     reader = csv.reader(f)
#     data = []  # type: List[List]
#     labels = []  # type: List[int]
#     records = []
#     count = 0
#
#     start = time.time()
#     for row in reader:
#         if count == 0:  # 过滤掉第一行的表头
#             count += 1
#         else:
#             sentence = row[0]  # 读取语句
#             sentence_vector = vector.get_sentence_vector(sentence)  # 获取语句的向量
#             backup = row[2]  # 获取该文件对应的备份策略
#
#             # 将备份策略转换成标签
#             if backup in records:
#                 labels.append(records.index(backup))  # 装入标签
#             else:
#                 labels.append(len(records))  # 装入标签
#                 records.append(backup)
#
#             data.append(sentence_vector)  # 装入数据点
#
#             count += 1
#     end = time.time()
#     print(end - start)