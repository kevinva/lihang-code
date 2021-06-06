import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, w_size):
        self.w = np.ones(w_size, dtype=np.float32)
        self.b = 0
        self.learning_rate = 0.01
        self.loss_list = []

        print('W shape: ', self.w.shape)

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y
    
    def fit(self, X_train, y_train):
        is_wrong = True
        while is_wrong:
            wrong_count = 0
            loss = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                loss_temp = y * self.sign(X, self.w, self.b)
                if loss_temp <= 0:
                    loss += loss_temp
                    self.w = self.w  + self.learning_rate * np.dot(y, X)
                    self.b = self.b + self.learning_rate * y
                    wrong_count += 1
                if wrong_count == 0:
                    is_wrong = False
            print("loss: ", loss)
            self.loss_list.append(-loss)
        print('Finish!')
        return 'Perceptron Model!'
    
    def plotLoss(self):
        x = np.arange(len(self.loss_list))
        plt.plot(x, self.loss_list, label='train')
        plt.ylabel('loss')
        plt.show()

    def score(self):
        pass

