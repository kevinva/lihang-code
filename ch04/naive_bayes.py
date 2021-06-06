import numpy as np
import pandas as pd


class NB:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.classes = None
        self.prior = None
        self.class_prior = None
        self.class_count = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        # print(X)
        # print(y)

        self.class_count = y[y.columns[0]].value_counts()  # 每个类别的数目
        self.class_prior = self.class_count / y.shape[0]   # 先验概率 P(y = Ck)

        # print(self.class_count)
        # print(self.class_prior)
        
        self.prior = dict()
        for idx in X.columns:
            for j in self.classes:
                # print((y == j).values)
                p_x_y = X[(y == j).values][idx].value_counts()  # X[(y == j).values]，提取分类为y == j的那些行
                # print(" hoho: ", X[(y == j).values])
                # print(p_x_y.index)
                # print('column: {}, \nclass: {}, \np_x_y={}'.format(idx, j, p_x_y))   # p_x_y 为当前类别下, 特征为x指定值的数量
                for i in p_x_y.index:
                    self.prior[(idx, i, j)] = p_x_y[i] / self.class_count[j]   # 条件概率P(X = i | y = j)

        # print(self.prior)

    def predict(self, X):
        rst = []
        for class_ in self.classes:
            py = self.class_prior[class_]   # 先验概率P(y = Ck)
            pxy = 1
            for idx, x in enumerate(X):
                pxy *= self.prior[(idx, x, class_)]   # 计算 P(X = x1 | y = c1) * P(X = x2 | y = c1) * ... * P(X = xn | y = c1)
            
            rst.append(py * pxy)   # py * pxy 为后验概率的分子部分
        return self.classes[np.argmax(rst)]