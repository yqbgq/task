import numpy as np


def cal_dis(data: np.ndarray, use_matrix=True):
    """
    计算每个样本到其他样本之间的距离

    :param use_matrix:  是否使用矩阵计算，使用矩阵计算可以避免循环，但是会将数据矩阵的大小扩大行数大小倍，导致内存溢出
    :param data: 样本集合 M*N M 个样本，每个样本有 N 个特征
    :return: 距离矩阵 N*N
    """
    if use_matrix:
        if use_matrix:
            g = np.dot(data, data.T)
            g_dia = np.diag(g)
            n = data.shape[0]
            d = np.zeros([n, n])
            for i in range(n):
                d[i, :] = g_dia[i] - 2 * g[i, :] + g_dia
            return d ** (1 / 2)
    else:
        dis_array = np.ndarray([data.shape[0], data.shape[0]])
        for i in range(data.shape[0]):
            dis = np.sum((data - data[i]) ** 2, axis=1) ** (1 / 2)
            dis_array[i] = dis

        return dis_array
