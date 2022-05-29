import numpy as np
import matplotlib.pyplot as plt

class gradientDescent:
    def __init__(self, mu=0.3, gamma=0.8, eta=0.016, epsilon=0.000001, x1=1, x2=0):
        self.mu = mu
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.x = np.array([[x1], [x2]])
        print('初始值为%d, %d'%(x1, x2))
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        d = np.array([[0], [0]])
        while flag:
            x1 = x[0, 0] + self.gamma*d[0, 0]
            x2 = x[1, 0] + self.gamma*d[1, 0]
            
            d1 = 0.2*x1
            d2 = 4*x2
            
            f_der = np.array([[d1], [d2]])
            if np.linalg.norm(f_der) < self.epsilon:
                flag = False

            d = self.gamma*d-self.eta*f_der
            index += 1
            # print('第%d次迭代:'%(index))
            # print(x)
            # print(alpha)
            # print(d)
            x = x + d
        print('经历%d次迭代后收敛'%(index))
        return x
    def draw(self):
        plt.figure(figsize=(6,6))
        plt.contour(X, Y, Z, 50)
        xx = [a[0][0] for a in xs]
        yy = [a[1][0] for a in xs]
        l = len(xx)
        nums = 100
        step = l//nums
        idxs = [i for i in range(0,l,step)]
        if l%step != 0:idxs.append(l-1)
        x_n = [xx[i] for i in idxs]
        y_n = [yy[i] for i in idxs]
        plt.plot(x_n,y_n,color='r',marker='o',markerfacecolor='blue',markersize=12)
        plt.show()
    
    
    
if __name__ == '__main__':
    a = gradientDescent()
    x = a.solve()
    print(x)