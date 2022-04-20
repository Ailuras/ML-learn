import numpy as np

class newtonMethod:
    def __init__(self, epsilon=0.001, x1=10, x2=10):
        self.epsilon = epsilon
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
            
            x = x - np.dot(np.matrix(f_der2).I, f_der1)
            x1 = x[0, 0]
            x2 = x[1, 0]
            f2 = x1**2 + 3*x2**2 - 4*x1 - 2*x1*x2
            
            if abs(f1-f2) < self.epsilon:
                flag = False
            index += 1
            # print('第%d次迭代:'%(index))
            # print(x)
            # print(alpha)
            # print(d)
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)