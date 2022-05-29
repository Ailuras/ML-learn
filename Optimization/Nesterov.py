import numpy as np
import matplotlib.pyplot as plt

class gradientDescent:
    def __init__(self, mu=0.3, gamma=0.9, eta=0.1, epsilon=0.001, x1=1, x2=1):
        self.mu = mu
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.x1 = []
        self.x2 = []
        self.x = np.array([[x1], [x2]])
        print('初始值为%d, %d'%(x1, x2))
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        d = np.array([[0], [0]])
        while flag:
            self.x1.append(x[0, 0])
            self.x2.append(x[1, 0])
            x1 = x[0, 0] + self.gamma*d[0, 0]
            x2 = x[1, 0] + self.gamma*d[1, 0]
            d1 = 0.2*x1
            d2 = 4*x2
            
            f_der = np.array([[d1], [d2]])
            if np.linalg.norm(f_der) < self.epsilon:
                flag = False
            d = self.gamma*d-self.eta*f_der
            index += 1
            x = x + d
        print('经历%d次迭代后收敛'%(index))
        return x
    def draw(self):
        step = 0.5
        x = np.arange(-2, 2, step)
        y = np.arange(-2, 2, step)
        X, Y = np.meshgrid(x, y)
        Z = 0.01*X**2 + 2*Y**2
        plt.figure(figsize=(6, 6))
        plt.contour(X, Y, Z, 50)
        plt.plot(self.x1, self.x2)
        plt.scatter(self.x1, self.x2, color='r', s=3)
        plt.title('gamma = 0.9, eta = 0.1')
        plt.show()
    
    
    
if __name__ == '__main__':
    a = gradientDescent()
    x = a.solve()
    print(x)
    a.draw()