from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
import time


class LogisticRegression:
    
    def __init__(self, learning_step=0.0001, epsilon=0.001, n_iter=1500):
        self.learning_step = learning_step
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.coef = np.array([])
        self.cols = []

    def fit(self, x, y):
        return self.gradient_descent(x, y, epsilon=self.epsilon, n_iter=self.n_iter)

    def predict(self, x):
        # print('x shape: ', x.shape)
        # print('w shape: ', self.coef.shape)
        result = np.array([self.cols[idx] for idx in [np.argmax(rst) for rst in sigmoid(np.dot(x, self.coef.T))]])
        return result
    
    def gradient_descent(self, x, y, epsilon=0.00001, n_iter=1500):
        n = x.shape[len(x.shape) - 1]
        y = pd.get_dummies(y)  # one-hot encoding
        w = np.array([])
        # print(n, y.shape, y.columns)
        # print(y)

        for ck in np.arange(y.shape[1]):
            wck = np.zeros(n)
            for k in np.arange(n_iter):
                g_k = self.g(x, y.values[:, ck], wck)  # 计算权重误差（求导）

                if np.average(g_k * g_k) < epsilon:    # 梯度裁剪：梯度小于一定的阈值，则不再更新权重
                    w = wck if w.size == 0 else np.vstack([w, wck])
                    break
                else:
                    p_k = -g_k
                lambda_k = 0.0001  # 学习速率
                wck = wck + lambda_k * p_k
            if k == n_iter - 1:
                w = wck if w.size == 0 else np.vstack([w, wck])
            print('progress: %d done' % ck)
        self.coef = w
        self.cols = y.columns.tolist()
        return self.coef, self.cols

def sigmoid(x):
    p = np.exp(x)
    p = p / (1 + p)
    return p

# 逻辑回归损失函数Loss(w)对w求偏导 
def g(x, y, w):
    m = y.size
    result = - (1 / m) * np.dot(x.T, y - sigmoid(np.dot(x, w)))
    return result

def load_data(path='./train.csv'):
    raw_data = pd.read_csv(path)
    y = raw_data['label'].values
    del raw_data['label']
    X = raw_data.values
    return X, y


if __name__ == '__main__':
    # x = np.random.randn(3, 10)
    # w = np.zeros(10)
    # print(w)
    # print(w.shape)
    # print(w.size)
    # print(np.dot(x, w).shape)

    print('Start reading data...')
    time_1 = time.time()
    X, y = load_data()
    print(X.shape)
    X = X[:200]
    y = y[:200]
    # print(X.shape)
    # print(y.shape)
    # y_test = pd.get_dummies(y)
    # print(y_test.shape)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2021)
    # print(set(train_y), set(test_y))
    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' seconds.\n')

    print('Start training...')
    clf = LogisticRegression()
    # clf.f = f
    clf.g = g
    clf.fit(train_x, train_y)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' seconds.\n')

    print('Start predicting...')
    test_predict = clf.predict(test_x)
    time_4 = time.time()
    print('predicting cose ', time_4 - time_3, ' seconds.\n')

    score = accuracy_score(test_y, test_predict)
    print('The accuracy score is ', score)
