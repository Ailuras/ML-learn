import numpy as np

class newtonMethod:
    def __init__(self, epsilon=1e-10, n=7):
        self.epsilon = epsilon
        self.n = n
        self.x = np.array([0.0 for i in range(n)]).reshape((-1,1))
        print('n为%d'%(n))
    
    def solve(self):
        x = self.x
        flag = True
        index = 0
        while flag:
            f1 = 0.0
            for i in range(self.n-1):
                f1 += (1-x[i, 0])**2 + 100*(x[i+1, 0]-x[i, 0]**2)**2
            
            f_der1 = np.array([0.0 for _ in range(self.n)]).reshape((-1,1))
            for i in range(self.n-1):
                f_der1[i, 0] += 2*(x[i, 0]-1) - 400*x[i, 0]*(x[i+1, 0]-x[i, 0]**2)
                f_der1[i+1, 0] += 200*(x[i+1, 0]-x[i, 0]**2)

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
            
            if np.linalg.norm(f_der1) < self.epsilon:
                flag = False
            index += 1
        print('经历%d次迭代后收敛'%(index))
        return x

if __name__ == '__main__':
    a = newtonMethod()
    x = a.solve()
    print(x)