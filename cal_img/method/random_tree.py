from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as sm

from utils.load_data import dataset


datasets = dataset()

train_samples, train_labels = datasets.exclude(0, [])
test_samples, test_labels = datasets.load_test_data()

tree = RandomForestClassifier(max_depth=10, n_estimators=100, min_samples_split=3)
tree.fit(train_samples, train_labels)

print(tree.feature_importances_)

result = tree.predict(test_samples)

print('得分：', sm.r2_score(test_labels, result))