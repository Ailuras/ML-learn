import numpy as np

class newtonMethod:
    def __init__(self, epsilon=0.000001, x1=1, x2=0):
        self.epsilon = epsilon
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
            
            d11 = 30*x1**4 - 12*x1*x2 + 24*(x2-x1)**2
            d12 = -6*x1**2 - 24*(x2-x1)**2
            d21 = -6*x1**2 - 24*(x2-x1)**2
            d22 = 2 + 24*(x2-x1)**2
            f_der2 = np.array([[d11, d12], [d21, d22]])
            # print(f_der2)
            x = x - np.dot(np.matrix(f_der2).I, f_der1)
            x1 = x[0, 0]
            x2 = x[1, 0]
            f2 = (x1**3-x2)**2 + 2*(x2-x1)**4
            
            if np.linalg.norm(f_der1) < self.epsilon:
                flag = False
            index += 1
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)