import numpy as np
import matplotlib.pyplot as plt

class Momentum:
    def __init__(self, mu=0.3, gamma=0.8, eta=0.002, epsilon=0.01, n=2):
        self.mu = mu
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.n = n
        self.y = []
        self.d = []
        self.x1 = []
        self.x2 = []
        self.x = np.array([0.0 for i in range(n)]).reshape((-1,1))
        print('n为%d'%(n))
    
    def solve(self):
        x = self.x
        flag = True
        self.index = 0
        d = np.array([0.0 for _ in range(self.n)]).reshape((-1,1))
        while flag:
            f1 = 0.0
            for i in range(self.n-1):
                f1 += (1-x[i, 0])**2 + 100*(x[i+1, 0]-x[i, 0]**2)**2   
            f_der = np.array([0.0 for _ in range(self.n)]).reshape((-1,1))
            for i in range(self.n-1):
                f_der[i, 0] += 2*(x[i, 0]-1) - 400*x[i, 0]*(x[i+1, 0]-x[i, 0]**2)
                f_der[i+1, 0] += 200*(x[i+1, 0]-x[i, 0]**2)
            self.index += 1
            self.y.append(f1)
            self.d.append(np.linalg.norm(f_der))
            if self.n == 2:
                self.x1.append(x[0, 0])
                self.x2.append(x[1, 0])
            if np.linalg.norm(f_der) < self.epsilon:
                flag = False
                                
            d = self.gamma*d-self.eta*f_der
            x = x + d
        print('经历%d次迭代后收敛'%(self.index))
        return x

    # 用于第三题画图
    def show(self):
        x = range(self.index)
        y = self.y
        plt.title('Train_Loss_list')
        plt.plot(x, y, label="Train_Loss_list")
        plt.show()
        x = range(self.index)
        y = self.d
        plt.title('Train_gradient_list')
        plt.plot(x, y, label="Train_gradient_list")
        plt.show()
    
    # 用于第二题画图
    def draw(self):
        step = 0.5
        x = np.arange(0, 2, step)
        y = np.arange(0, 2, step)
        X, Y = np.meshgrid(x, y)
        Z = (1-X)**2 + 100*(Y-X**2)**2
        plt.figure(figsize=(6, 6))
        plt.contour(X, Y, Z, 50)
        
        plt.plot(self.x1, self.x2)
        plt.scatter(self.x1, self.x2, color='r', s=3)
        plt.show()

if __name__ == '__main__':
    a = Momentum()
    x = a.solve()
    print(x)
    a.draw()