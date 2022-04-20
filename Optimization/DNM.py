import numpy as np

class newtonMethod:
    def __init__(self, mu=0.3, beta=0.8, epsilon=0.001, alpha=1, x1=1, x2=0):
        self.mu = mu
        self.beta1 = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.x = np.array([[x1], [x2]])
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        while flag:
            x1 = x[0, 0]
            x2 = x[1, 0]
            f1 = x1**2 + 3*x2**2 - 4*x1 - 2*x1*x2
            
            d1 = 2*x1 - 4 - 2*x2
            d2 = 6*x2 - 2*x1
            f_der1 = np.array([[d1], [d2]])
            
            d11 = 2
            d12 = -2
            d21 = -2
            d22 = 6
            f_der2 = np.array([[d11, d12], [d21, d22]])
            
            d = -np.dot(np.matrix(f_der2).I, f_der1)
            
            alpha = self.alpha
            temp = np.dot(f_der1.T, d)
            while True:
                y = x + alpha*d
                y1 = y[0, 0]
                y2 = y[1, 0]
                f2 = y1**2 + 3*y2**2 - 4*y1 - 2*y1*y2
                if -alpha*temp*self.mu > f1-f2:
                    alpha = alpha*self.beta
                    continue
                # elif temp*(1-self.mu) < f1-f2:
                #     print('太小了', alpha)
                #     alpha = alpha*self.beta2
                #     continue
                else:
                    if abs(f1-f2) < self.epsilon:
                        flag = False
                    break
                
            index += 1
            # print('第%d次迭代:'%(index))
            # print(x)
            # print(alpha)
            # print(d)
            x = x + alpha*d
            # break
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)