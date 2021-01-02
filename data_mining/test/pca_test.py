from sklearn.decomposition import PCA
from data_mining.utils import read_data

data, labels = read_data.get_data()
pca = PCA(n_components= 25)
reduced_x = pca.fit_transform(data)
print("finish")