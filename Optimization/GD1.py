import numpy as np

class gradientDescent:
    def __init__(self, mu=0.3, beta1=0.8, beta2=1.5, epsilon=0.000001, alpha=1, x1=1, x2=0):
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.x = np.array([[x1], [x2]])
        print('初始值为%d, %d'%(x1, x2))
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        while flag:
            x1 = x[0, 0]
            x2 = x[1, 0]
            f1 = (x1**3-x2)**2 + 2*(x2-x1)**4
            
            d1 = 6*(x1**3-x2)*x1**2 - 8*(x2-x1)**3
            d2 = -2*(x1**3-x2) + 8*(x2-x1)**3
            
            f_der = np.array([[d1], [d2]])
            if np.linalg.norm(f_der) < self.epsilon:
                flag = False

            d = -f_der
            alpha = self.alpha
            while True:
                y = x + alpha*d
                y1 = y[0, 0]
                y2 = y[1, 0]
                f2 = (y1**3-y2)**2 + 2*(y2-y1)**4
                
                temp = -alpha*np.dot(f_der.T, d)
                if temp*self.mu > f1-f2:
                    # print(alpha)
                    alpha = alpha*self.beta1
                    continue
                elif temp*(1-self.mu) < f1-f2:
                    # print(alpha)
                    alpha = alpha*self.beta2
                    continue
                else:
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