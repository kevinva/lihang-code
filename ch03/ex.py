import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from knn import KNN
from kdtree import *
from random import random

# def L(x, y, p=2):
#     if len(x) == len(y) and len(x) > 1:
#         sum = 0
#         for i in range(len(x)):
#             sum += math.pow(abs(x[i] - y[i]), p)
#         return math.pow(sum, 1.0 / p)
#     else:
#         return 0

# x1 = [1, 1]
# x2 = [5, 1]
# x3 = [4, 4]

# for i in range(1, 5):
#     r = {'1-{}'.format(c) : L(x1, c, p=i) for c in [x2, x3]}
#     print(min(zip(r.values(), r.keys())))


# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['label'] = iris.target
# df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# print(df.head())

# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

# data = np.array(df.iloc[:100, [0, 1, -1]])  # 取第0列，第1列，和最后一列
# X, y = data[:, :-1], data[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# knn_clf = KNN(X_train, y_train, n_neighbors=12)
# print(knn_clf.score(X_test, y_test))

# test_point = [6.0, 3.0]
# print('Test point: {}'.format(knn_clf.predict(test_point)))

data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
preorder(kd.root)

def random_point(k):
    return [random() for _ in range(k)]

def random_points(k, n):
    return [random_point(k) for _ in range(n)]

    
ret = find_nearest(kd, [3, 4.5])
print(ret)