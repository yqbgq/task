# import copy
# import inspect
# import math
# import sys
# import time
# import threading
#
# import numpy as np
# import tqdm
#
# sys.path.append('..')
#
# from script import preprocess_img
# from utils import write_log
# from method import multi_class_svm
# from utils.load_data import get_data
# from utils import read_ini
# from utils.print_info import print_info
#
# import ctypes
#
# features = ["color_data", "his_data", "sobel_data"]
#
#
# def _async_raise(tid, exctype):
#     """raises the exception, performs cleanup if needed"""
#     tid = ctypes.c_long(tid)
#     if not inspect.isclass(exctype):
#         exctype = type(exctype)
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
#     if res == 0:
#         raise ValueError("invalid thread id")
#     elif res != 1:
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")
#
#
# def stop_thread(thread):
#     _async_raise(thread.ident, SystemExit)
#
#
# class ProcessThreading(threading.Thread):
#     def __init__(self, pars):
#         super(ProcessThreading, self).__init__()
#         self.pars = pars
#         self.acc = 0
#         self.clf_list = []
#
#     def run(self):
#         degree, coef0, c = self.pars
#         read_ini.config.set_par("Train-Par", "degree", degree)
#         read_ini.config.set_par("Train-Par", "coef0", coef0)
#         read_ini.config.set_par("Train-Par", "C", c)
#
#         self.clf_list, self.acc = assemble_learning()
#
#     def get_result(self):
#         return self.clf_list, self.acc
#
#
# def train():
#     max_acc = 0
#     for degree in range(3, 10):
#         for coef0 in range(20, 100):
#             with tqdm.trange(20, 150) as t:
#                 for c in t:
#                     pars = read_ini.config.get_all()
#                     clf_list, acc = [], 0
#                     p = ProcessThreading([degree, coef0, c])
#                     p.start()
#                     start = time.time()
#                     while time.time() - start < 125:
#                         clf_list, acc = p.get_result()
#                         if acc == 0:
#                             continue
#                         else:
#                             break
#                     des = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
#                     if acc == 0:
#                         try:
#                             stop_thread(p)
#                         except (ValueError, SystemError):
#                             time.sleep(20)
#                             write_log.log.log(des, "关闭了一个进程！！！！")
#                     else:
#
#                         post = "deg {:.2f}, coef0 {:.2f}, C {:.2f}, ACC {:.2f}% clf {}".format(
#                             degree, coef0, c, acc * 100, len(clf_list)
#                         )
#
#                         if acc > max_acc:
#                             first = pars
#                             second = "above acc {:.2f}%, clf num :{}".format(acc * 100, len(clf_list))
#                             write_log.log.log(first, second)
#                             max_acc = acc
#                     t.set_description_str(des)
#                     t.set_postfix_str(post)
#
#
# def get_weight(clf, al_train_data, al_train_label):
#     acc = multi_class_svm.test(al_train_data, al_train_label, clf)
#     weight = 0.5 * math.log(acc / (1 - acc), 10)
#     return weight
#
#
# def get_total_acc(clf_list, data, label):
#     result = np.ndarray([0])
#     for i in range(len(clf_list)):
#         if i == 0:
#             result = clf_list[i][1].predict_proba(data) * clf_list[i][0]
#         else:
#             result = result + clf_list[i][1].predict_proba(data) * clf_list[i][0]
#     result = result
#     result = np.argmax(result, axis=1)
#     check = (result == label)
#     acc = np.sum(check) / label.shape[0]
#     return acc, check
#
#
# def reallocate(clf_list, al_train_data, al_train_label):
#     _, check = get_total_acc(clf_list, al_train_data, al_train_label)
#
#     false_idx = np.argwhere(check == False).reshape(-1)
#     false_sample = al_train_data[false_idx]
#     false_label = al_train_label[false_idx]
#     al_train_data = np.vstack([al_train_data, false_sample])
#     al_train_label = np.hstack([al_train_label, false_label])
#     drop_idx = np.random.choice(np.arange(al_train_data.shape[0]), size=false_idx.shape[0] * 2, replace=False)
#     al_train_data = np.delete(al_train_data, drop_idx, axis=0)
#     al_train_label = np.delete(al_train_label, drop_idx, axis=0)
#     return al_train_data, al_train_label
#
#
# def assemble_learning():
#     last_acc = 0
#     clf_list = []
#     al_train_data, al_train_label = copy.copy(train_data), copy.copy(train_label)
#     while True:
#         clf = multi_class_svm.train(al_train_data, al_train_label)
#         weight = get_weight(clf, al_train_data, al_train_label)
#         clf_list.append([weight, clf])
#         total_acc, _ = get_total_acc(clf_list, test_data, test_label)
#         if (total_acc <= last_acc or abs(total_acc - last_acc) < 0.05) and len(clf_list) >= 3:
#             clf_list.pop()
#             return clf_list, last_acc
#         else:
#             last_acc = total_acc
#             al_train_data, al_train_label = reallocate(clf_list, al_train_data, al_train_label)
#
#
# if __name__ == "__main__":
#     print_info("开始进行图片切割")
#
#     # preprocess_img.preprocess_img()  # 根据训练的参数来进行图片的预处理
#
#     print_info("图片切割完成，开始训练")
#     # features = ["color_data", "his_data", "sift_data"]
#
#     print_info("开始加载训练数据集")
#     train_data, train_label = get_data(feature=features, training=True)
#     test_data, test_label = get_data(feature=features, training=False)
#
#     train()
