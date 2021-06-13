import math
from copy import deepcopy

from numpy import mat


class MaxEntropy:
    
    def __init__(self, EPS=0.005):
        self._samples = []
        self._Y = set()
        self._numXY = {}
        self._N = 0   # 样本数
        self._Ep_ = []   # 样本分布的特征值期望
        self._xyID = {}
        self._n = 0  # 特征键值（x, y）的个数
        self._C = 0  # 最大特征数
        self._IDxy = {}
        self._w = []
        self._EPS = EPS
        self._lastw = []

    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)
            for x in X:
                if (x, y) in self._numXY:
                    self._numXY[(x, y)] += 1
                else:
                    self._numXY[(x, y)] = 1
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in self._samples])
        self._w = [0] * self._n
        self._lastw = self._w[:]

        self._Ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):   # 计算特征函数f_i 关于经验分布的期望
            self._Ep_[i] = self._numXY[xy] / self._N  # hoho_todo，期望是这样算的？
            self._xyID[xy] = i
            self._IDxy[i] = xy
    
    def Zx(self, X):   # 计算每个Z(x)值
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def model_pyx(self, y, X):   # 计算每个P(y|x)
        zx = self.Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / zx            # hoho_todo, 为啥要加exp(...)
        return pyx

    def model_ep(self, index):    # 计算特征函数f_i 关于模型的期望
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self.model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def convergence(self):
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._EPS:
                return False
        return True

    def train(self, maxiter=1000):
        for loop in range(maxiter):
            print('iter: %d' % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self.model_ep(i)
                self._w[i] += math.log(self._Ep_[i] / ep) / self._C
            print('w:', self._w)
            if self.convergence():
                break

    def predict(self, X):
        Z = self.Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result


if __name__ == '__main__':
    dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]
    maxent = MaxEntropy()
    maxent.loadData(dataset)
    maxent.train()

    x = ['overcast', 'mild', 'high', 'FALSE']
    print('predict: ', maxent.predict(x))