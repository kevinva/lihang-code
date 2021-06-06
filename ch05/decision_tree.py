from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DecisionTree:

    def __init__(self, eps, feas, name=None, criterion='entropy'):
        self.tree = dict()
        self.features = feas
        self.eps = eps
        self.criterion = criterion
        if not name:
            self.name = "Decision Tree"
        else:
            self.name = name
        
    def fit(self, x, y):
        self.tree = self.build_tree(x, y, self.eps)
        return self.tree
    
    def predict(self, x, x_tree=None):
        pass

    def build_tree(self, x, y, eps):
        features = np.arange(x.shape[1])
        labels = y

        if len(set(labels)) == 1:
            return labels[0]


        # 数量最多的类别
        max_label = max([(i, len(list(filter(lambda tmp: tmp == i, labels)))) for i in set(labels)], key=lambda tmp: tmp[1])[0]

        if len(features) == 0:
            return max_label

        max_feature = 0
        max_criterion = 0
        D = labels
        for feature in features:
            A = x[:, feature]
            if self.criterion == 'entropy':
                gda = gain(A, D)
            elif self.criterion == 'gr':
                gda = gain_ratio(A, D)
            elif self.criterion == 'gini':
                pass
                
            if max_criterion < gda:
                max_criterion, max_feature = gda, feature

# 计算熵
def cal_ent(x):
    x_values = list(set(x))
    ent = 0
    for x_value in x_values:
        p = x[x == x_value].shape[0] / x.shape[0]
        ent -= p * np.log2(p)
    return ent

# 计算条件熵 ent(y | x)
def cal_condition_ent(x, y):
    ent = 0
    x_values = set(x)
    for x_value in x_values:
        sub_y = y[x == x_value]
        tmp_ent = cal_ent(sub_y)
        p = sub_y.shape[0] / y.shape[0]
        ent += p * tmp_ent
    return ent

# 信息增益
def gain(x, y):
    return cal_ent(y) - cal_condition_ent(x, y)

# 信息增益比
def gain_ratio(x, y):
    return gain(x, y) / cal_ent(x)
            
                
            
