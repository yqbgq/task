from utils.load_data import dataset

datasets = dataset()
data, labels = datasets.load_test_data()

print("ok")