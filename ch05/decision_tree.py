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
    
    def predict(self, x, clf, x_tree=None):
        if len(x.shape) == 2:
            rst = []
            for x_ in x:
                rst.append(self.predict(x_))
            return rst

        if not x_tree:
            x_tree = self.tree
        tree_key = list(x_tree.keys())[0]
        x_feature = tree_key.split('__')[0]
        x_idx = clf.features.index(x_feature)
        x_tree = x_tree[tree_key]
        for key in x_tree.keys():
            if key.split('__')[0] == x[x_idx]:
                tree_key = key
                x_tree = x_tree[tree_key]
        if type(x_tree) == dict:
            return self.predict(x, clf, x_tree)
        else:
            return x_tree

    def build_tree(self, x, y, eps):
        features = np.arange(x.shape[1])
        labels = y

        # 如果该节点只有一个类别，则选该类别
        if len(set(labels)) == 1:
            return labels[0]

        # 数量最多的类别
        max_label = max([(i, len(list(filter(lambda tmp: tmp == i, labels)))) for i in set(labels)], key=lambda tmp: tmp[1])[0]

        # 如果该节点无更多的可划分的特征，则选数量最多的类别
        if len(features) == 0:
            return max_label

        max_feature = 0   # 用于划分的最优特征
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

        # 信息增益大于一定的阈值，则直接选该节点数量最多的类别
        if max_criterion < eps:
            return max_label
        
        T = dict()
        sub_T = dict()
        for x_A in set(x[:, max_feature]):
            sub_D = D[x[:, max_feature] == x_A]     # 属于相同特征值的类别列表
            sub_x = x[x[:, max_feature] == x_A, :]  # 属于相同特征值的样本
            sub_x = np.delete(sub_x, max_feature, 1) # 删除max_feature对于特征的那一列，第三个参数表示axis
            sub_T[str(x_A) + '__' + str(sub_D.shape[0])] = self.build_tree(sub_x, sub_D, eps)   # 以“特征值__对应类别数”作为key
        
        T[str(self.features[max_feature]) + '__' + str(D.shape[0])] = sub_T
        return T

        ### hoho_check!



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
            
                
            
