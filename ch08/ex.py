import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adaboost import AdaBoost

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1

    return data[:, :2], data[:, -1]


X, y = create_data()
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# plt.scatter(X[:50, 0], X[:50, 1], label='0')
# plt.scatter(X[50:, 0], X[50:, 1], label='1')
# plt.legend()
# plt.show()


clf = AdaBoost(n_estimators=50, learning_rate=0.2)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
