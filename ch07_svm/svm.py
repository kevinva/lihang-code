import numpy as np
import time

class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        self.C = 1.0

    def _KKT(self, i):  # hoho_todo，what? (根据KTT条件得到的，P149)
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1
        
    def _g(self, i):   # 分类决策函数
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r

    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
        return 0

    def _E(self, i):   # 样本的误差值
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]   # 0 < self.alpha[i] < self.C 为间隔边界上的支持向量点，优先遍历它们，因为它们更需要调整
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue
            
            E1 = self.E[i]   # 找到违反KTT条件最严重的第一个变量alpha_i
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x]) # 选择的alpha_j 要满足|E_i - E_j|最大，所以，如果E_i是正的，选最小的E_j可保证|E_i - E_j|最大
            else:
                j = max(range(self.m), key=lambda x: self.E[x]) # 如果E_i是负的，则选最大的E_j可保证|E_i - E_j|最大

            return i, j    # 找到误差最大的第二个变量alpha_j

    def _compare(self, _alpha, L, H):  # 优化的时候，alpha要满足的约束
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):   # 使用SMO算法
        self.init_args(features, labels)

        for t in range(self.max_iter):
            start_time = time.time()

            i1, i2 = self._init_alpha()  # 选择两个alpha变量

            # 可有关于两个alpha变量的二维坐标图得出变量的可行域
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)  # 大于最大的下界
                H = min(self.C, self.alpha[i1] + self.alpha[i2])   # 小于最小的上界
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            # hoho_todo: eta? what? (参考书 P145~P146，表示两样本的相似性)
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)  # 参数书 P145~P146
            
            # b值的计算，参考书 P148
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

            end_time = time.time()
            print('iter {}, elasp: {}\n'.format(t, end_time - start_time))
        
        return "train done!"

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w