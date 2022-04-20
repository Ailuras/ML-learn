import numpy as np

class gradientDescent:
    def __init__(self, mu=0.3, beta1=0.8, beta2=1.5, epsilon=0.0001, alpha=1, x1=4, x2=4):
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.x = np.array([[x1], [x2]])
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        while flag:
        # for i in range(10):
            x1 = x[0, 0]
            x2 = x[1, 0]
            f1 = x1**2 + 3*x2**2 - 4*x1 - 2*x1*x2
            # f1 = x1*x1 + x2*x2
            
            d1 = 2*x1 - 4 - 2*x2
            d2 = 6*x2 - 2*x1
            # d1 = 2*x1
            # d2 = 2*x2
            
            f_der = np.array([[d1], [d2]])
            d = -f_der
            alpha = self.alpha
            while True:
                y = x + alpha*d
                # print(y)
                y1 = y[0, 0]
                y2 = y[1, 0]
                f2 = y1**2 + 3*y2**2 - 4*y1 - 2*y1*y2
                # f2 = y1*y1 + y2*y2
                
                temp = -alpha*np.dot(f_der.T, d)
                if temp*self.mu > f1-f2:
                    alpha = alpha*self.beta1
                    continue
                elif temp*(1-self.mu) < f1-f2:
                    alpha = alpha*self.beta2
                    continue
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
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = gradientDescent()
    x = a.solve()
    print(x)