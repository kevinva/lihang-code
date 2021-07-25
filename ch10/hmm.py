import numpy as np

class HiddenMarkov:

    def __init__(self):
        self.alphas = None
        self.forward_P = None
        self.betas = None
        self.backward_P = None
    
    # A为状态转移矩阵，B为观测概率矩阵，PI为初始状态概率向量
    # 前向算法
    def forward(self, Q, V, A, B, O, PI):
        N = len(Q)  # 状态序列大小
        M = len(O)  # 观测序列大小
        alphas = np.zeros((N, M))
        T = M
        for t in range(T):
            indexOfO = V.index(O[t]) # V是啥
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]
                    print('alpha1 (%d) = p%d * b%db(o1) = %f' % (i + 1, i, i, alphas[i][t]))
                    
                else:
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas], [a[i] for a in A]) * B[i][indexOfO]
                    print('alpha%d (%d) = [sigma alpha%d(i) * ai%d] * b%d(o%d) = %f' % (t + 1, i, t, i, i, t, alphas[i][t]))
                    
        self.forward_P = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas
        
    
    # 后向算法
    def backward(self, Q, V, A, B, O, PI):
        N = len(Q)
        M = len(O)
        betas = np.ones((N, M))
        
        for i in range(N):
            print('beta%d(%d) = 1' % (M, i + 1))
            
        for t in range(M - 2, -1, -1):
            indexOfO = V.index(O[t + 1])
            
            for i in rnage(N):
                betas[i][t] = np.dot(np.multiply(A[i], [b[indexOfO] for b in B]), [beta[t + 1] for beta in betas])
                realT = t + 1
                realI = i + 1
                print('beta%d(%d) = sigma[a%dj * bj(o%d) * beta%d(j)] = (' %
                      (realT, realI, realI, realT + 1, realT + 1),
                      end='')
                      
                for j in range(N):
                    print("%.2f * %.2f * %.2f + " %
                          (A[i][j], B[j][indexOfO], betas[j][t + 1]),
                          end='')
                print("0) = %.3f" % betas[i][t])
                
        indexOfO = V.index(O[0])
        self.betas = betas
        p = np.dot(np.multiply(PI, [b[indexOfO] for b B]), [beta[0] for beta in betas])
        self.backward_P = P
        print('P(O | lambda) = ', end='')
        for i in range(N):
            print("%.1f * %.1f * %.5f + " % (PI[0][i], B[i][indexOfO], betas[i][0]), end="")
        print('0 = %f' % P)
        
    # 维特比算法（求最优状态序列）
    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q)  # 状态的值域列表
        M = len(O)
        deltas = np.zeros((N, M))
        psis = np.zeros((N, M))
        I = np.zeros((1, M))
        
        for t in range(M):
            realT = t + 1
            indexOfO = V.index(O[t])
            for i in range(N):
                realI = i + 1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f' %
                          (realI, realI, realI, PI[0][i], B[i][indexOfO],
                           deltas[i][t]))
                    print('psis1(%d) = 0' % (realI))
                else:
                    deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A])) * B[i][indexOfO]
                    print(
                        'delta%d(%d) = max[delta%d(j)aj%d]b%d(o%d) = %.2f * %.2f = %.5f'
                        % (realT, realI, realT - 1, realI, realI, realT,
                           np.max(
                               np.multiply([delta[t - 1] for delta in deltas],
                                           [a[i] for a in A])), B[i][indexOfO],
                           deltas[i][t]))
                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A]))
                    print('psis%d(%d) = argmax[delta%d(j)aj%d] = %d' %
                          (realT, realI, realT - 1, realI, psis[i][t]))
                          
        # 得到最优路径的终点
        I[0][M - 1] = np.argmax([delta[M - 1] for delta in deltas])
        print('i%d = argmax[deltaT(i)] = %d' % (M, I[0][M - 1] + 1))
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            print('i%d = psis%d(i%d) = %d' %
                  (t + 1, t + 2, t + 2, I[0][t] + 1))
        print('最优路径是：', '->'.join([str(int(i + 1)) for i in I[0]]))
                
