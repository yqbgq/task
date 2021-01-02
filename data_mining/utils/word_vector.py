"""
@Time 2020-12-5
@Author 黄伟
@Describe 词向量，从词向量文件中读取数据，并且支持词向量的查询以及句子向量的查询
"""

import numpy as np
import re



class vector:

    filter = re.compile("[^a-z^A-Z]")       # 过滤的正则表达式
    path = "./data/vector.txt"                 # 词向量地址
    data = {}                               # 词向量保存在 data 字典
    with open(path, encoding='utf-8') as file:
        line = file.readline()
        while line:
            temp = line.split(" ")
            vector = [float(x) for x in temp[1:]]
            data[temp[0]] = vector
            line = file.readline()
            
    keys = list(data.keys())                # 字典中键的列表，用于检查查询的单词是否存在字典中

    @staticmethod
    def get_word_vector(word):
        """
        静态方法，查询单词的向量

        :param word: 需要查询的向量
        :return:查询的向量值
        """
        return np.array(vector.data[word])

    @staticmethod
    def get_sentence_vector(sentence):
        """
        静态方法，查询句子的向量，计算比较简单，将语句中单词的向量求平均即可

        :param sentence: 需要查询的语句
        :return: 查询语句的向量值
        """
        result = np.zeros(50)
        sentence = vector.filter.sub(" ", sentence.strip()).lower()

        # 对数据进行清洗，去除单词列表中的空格，并且取前8个单词即可计算语句的向量
        word_list = [x for x in sentence.split(" ") if " " not in x and len(x) > 0][:6]
        for word in word_list:
            if word in vector.keys:
                temp = np.asarray(vector.data[word])
                result += temp

        return result//8
