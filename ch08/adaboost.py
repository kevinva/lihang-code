import numpy as np

class AdaBoost:

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape

        self.clf_sets = []

        self.weights = [1.0 / self.M] * self.M
        self.alpha = []   # G(x)的权重

    def _G(self, features, labels, weights):  # features 为各个样本的对应每个特征的列表
        m = len(features)
        error = 100000.0
        best_v = 0.0

        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            if v not in features:
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])  # 只计算错误分类的样本，赋予错误分类的样本更大的权重

                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]]) # 赋予错误分类的样本更大的权重

                # hoho_todo: 为啥不总体算一个错误值，而要区分正例负例各自的错误值取最大？
                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'
                
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
                    
        return best_v, direct, error, compare_array

    
    def _alpha(self, error):
        return 0.5 * np.log((1 - error + 1e-7) / (error + 1e-7))


    def _Z(self, weights, a, clf):
         return sum([weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])

    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1
    
    def fit(self, X, y):
        self.init_args(X, y)

        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None

            # 根据特征值维度，选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                
                if best_clf_error == 0:
                    break
            
            # 计算G(x)系数
            a = self._alpha(best_clf_error)
            self.alpha.append(a)

            self.clf_sets.append((axis, best_v, final_direct))
            Z = self._Z(self.weights, a, clf_result)
            self._w(a, clf_result, Z)

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

