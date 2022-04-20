import numpy as np

class newtonMethod:
    def __init__(self, mu=0.3, beta=0.8, epsilon=0.000001, alpha=1, x1=1, x2=0):
        self.mu = mu
        self.beta = beta
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
            f_der1 = np.array([[d1], [d2]])

            index += 1
            if np.linalg.norm(f_der1) < self.epsilon:
                flag = False
            
            d11 = 30*x1**4 - 12*x1*x2 + 24*(x2-x1)**2
            d12 = -6*x1**2 - 24*(x2-x1)**2
            d21 = -6*x1**2 - 24*(x2-x1)**2
            d22 = 2 + 24*(x2-x1)**2
            f_der2 = np.array([[d11, d12], [d21, d22]])
            
            d = -np.dot(np.matrix(f_der2).I, f_der1)
            
            alpha = self.alpha
            temp = np.dot(f_der1.T, d)
            while True:
                y = x + alpha*d
                y1 = y[0, 0]
                y2 = y[1, 0]
                f2 = (y1**3-y2)**2 + 2*(y2-y1)**4
                if -alpha*temp*self.mu > f1-f2:
                    alpha = alpha*self.beta
                    continue
                else:
                    break
            x = x + alpha*d
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)