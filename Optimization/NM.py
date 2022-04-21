import numpy as np
import matplotlib.pyplot as plt

class newtonMethod:
    def __init__(self, epsilon=1e-10, n=2):
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
        self.index = 0
        while True:
            f1 = 0.0
            for i in range(self.n-1):
                f1 += (1-x[i, 0])**2 + 100*(x[i+1, 0]-x[i, 0]**2)**2
            
            f_der1 = np.array([0.0 for _ in range(self.n)]).reshape((-1,1))
            for i in range(self.n-1):
                f_der1[i, 0] += 2*(x[i, 0]-1) - 400*x[i, 0]*(x[i+1, 0]-x[i, 0]**2)
                f_der1[i+1, 0] += 200*(x[i+1, 0]-x[i, 0]**2)

            self.index += 1
            self.y.append(f1)
            self.d.append(np.linalg.norm(f_der1))
            if self.n == 2:
                self.x1.append(x[0, 0])
                self.x2.append(x[1, 0])
            if np.linalg.norm(f_der1) < self.epsilon:
                break
            f_der2 = np.array([0.0 for _ in range(self.n**2)]).reshape((self.n, -1))
            for i in range(self.n-1):
                f_der2[i, i] += 2 - 400*x[i+1, 0] + 1200*x[i, 0]**2
                f_der2[i, i+1] += -400*x[i, 0]
                f_der2[i+1, i] += -400*x[i, 0]
                f_der2[i+1, i+1] += 200
            
            x = x - np.dot(np.matrix(f_der2).I, f_der1)
            f2 = 0.0
            for i in range(self.n-1):
                f2 += (1-x[i, 0])**2 + 100*(x[i+1, 0]-x[i, 0]**2)**2
        print('经历%d次迭代后收敛'%(self.index))
        return x
    
    def draw(self):
        def f(x, y):
            return (1-x)**2 + 100*(y-x**2)**2
        # plt.figure(figsize=(1000, 1000))
        n = 4096
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)

        X, Y = np.meshgrid(x, y)

        plt.contourf(X, Y, f(X, Y))
        plt.plot(self.x1, self.x2)
        plt.scatter(self.x1, self.x2, color='r', s=1)
        plt.show()
    
    def show(self):
        x = range(self.index)
        y = self.y
        plt.plot(x, y, label="Train_Loss_list")
        plt.show()
        x = range(self.index)
        y = self.d
        plt.plot(x, y, label="Train_Loss_list")
        plt.show()

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)
    # a.show()
    a.draw()