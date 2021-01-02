from data_mining.method import dbscan
from data_mining.method import hierarchical
from data_mining.method import iso_data
from data_mining.method import k_means

if __name__ == "__main__":
    while True:

        while True:
            data_type = input("请选择数据集\n 1：data.csv（癫痫发作识别数据） \n 2：issues.csv（控制图时间序列数据集）\n")
            if data_type == "2":
                issues = True
                break
            elif data_type == "1":
                issues = False
                break

        while True:
            method = input("请选择使用的分类器：\n"
                           "1：DBSCAN 算法\n"
                           "2：层次聚类算法\n"
                           "3：isodata 算法\n"
                           "4： k-means 算法\n")

            if method == "1":
                if issues:
                    instance = dbscan.dbscan(20, 2.0, issues)
                else:
                    instance = dbscan.dbscan(30, 1300, issues)  # 聚类准确率为
                instance.fit()
                break

            elif method == "2":
                if issues:
                    instance = hierarchical.hierarchical(32, issues=issues)
                else:
                    instance = hierarchical.hierarchical(2, issues=issues)
                instance.fit()
                break

            elif method == "3":
                if issues:
                    instance = iso_data.isodata(120, 10, 800, 400, 2, 200, issues=issues)
                else:
                    instance = iso_data.isodata(32, 20, 5, 2, 2, 200, issues=issues)
                instance.train()
                break

            elif method == "4":
                if issues:
                    instance = k_means.k_means(class_num=60, issues=issues)
                else:
                    instance = k_means.k_means(class_num=15, issues=issues)
                instance.train()
                break
