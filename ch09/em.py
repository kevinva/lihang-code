import math

class EM:

    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob  # A,B,C硬币正面出现的概率

    # E步，data是观测数据
    def pmf(self, data, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) *  math.pow((1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1 - self.pro_C), 1 - data[i])
        return pro_1 / (pro_1 + pro_2)

    
    def fit(self, data):
        count = len(data)
        print('init prob: {}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))

        for d in range(count):
            _ = yield
            _pmf = [self.pmf(data, k) for k in range(count)]
            pro_A= 1 / count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1 - _pmf[k]) * data[k] for k in range(count)]) / sum([(1 - _pmf[k]) for k in range(count)])

            print('{} / {} pro_a: {:.3f}, pro_b:{:.3f}, pro_C:{:.3f}'.format(d + 1, count, pro_A, pro_B, pro_C))

            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C

            

data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)
f.send(1)
f.send(2)
f.send(3) 