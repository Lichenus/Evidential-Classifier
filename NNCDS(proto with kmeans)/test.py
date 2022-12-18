from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
features = iris.data[...,2:]

model = KMeans(n_clusters=6)

model.fit(features)
F = [2,3]
c = model.cluster_centers_
print('centroids'+str(c))
# 选取行标为100的那条数据，进行预测
# prddicted_label = model.predict([[6.3, 3.3, 6, 2.5]])

x_axis = features[:, 0]
y_axis = features[:, 1]
# 预测全部150条数据
all_predictions = model.predict(features)


plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()
