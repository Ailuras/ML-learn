import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

class SGD:
    def __init__(self, dataFile='Data/wine/wine.data', beta=1000, alpha=0.01, gama=0.5, epsilon=0.0001, times=1500, bench_size=30):
        self.dataFile = dataFile
        self.data = []
        self.beta = beta
        self.alpha = alpha
        self.times = times
        self.gama = gama
        self.epsilon = epsilon
        self.bench_size = bench_size
        self.M = np.identity(13)
        self.get_info(self.dataFile)
    def get_info(self, dataFile):
        with open(dataFile) as f:
            line = f.readline()
            while line:
                data = line[:-1].split(',')
                self.data.append([float(i) for i in data])
                line = f.readline()
        data = np.array(self.data)
        for i in range(1, 14):
            data[:, i] = (data[:, i] - np.mean(data[:, i]))/ np.std(data[:, i])
        self.data = list(itertools.product(data, repeat=2))
        
    def get_error(self, M):
        error = 0
        for x_i, x_j in self.data:
            if x_i[0] == x_j[0]:
                temp = x_i[1:]-x_j[1:]
                temp = np.mat(temp)
                error += np.dot(np.dot(temp, M), temp.T)
        return error
    
    def solve(self):
        error = self.get_error(self.M)
        for _ in range(self.times):
            data = random.sample(self.data, self.bench_size)
            delta_M = np.zeros([13, 13])
            for x_i, x_j in data:
                temp = x_i[1:]-x_j[1:]
                temp = np.mat(temp)
                if x_i[0] == x_j[0]:
                    delta_M += np.dot(temp.T, temp)
                elif np.dot(np.dot(temp, self.M), temp.T) > 1:
                    delta_M += self.beta*np.dot(temp.T, temp)
                if np.dot(np.dot(temp, self.M), temp.T) < 0:
                    delta_M += self.beta*np.dot(temp.T, temp)
            
            alpha = self.alpha
            M_new = self.M-alpha*delta_M
            
            while self.get_error((M_new+M_new.T)/2) < 0:
                alpha *= self.gama
                M_new = self.M-alpha*delta_M
            self.M = (M_new+M_new.T)/2
            error = self.get_error(self.M)
            print(error)
            if error < self.epsilon:
                return
            print(np.linalg.eigvals(self.M))
            # if not (np.linalg.eigvals(self.M) >= 0).all():
            #     print('不正定')
    
    def test(self):
        self.data = []
        self.get_info('Data/wine/wine_test.data')
        # print(np.linalg.eigvals(self.M+self.M.T))
        # if not (np.linalg.eigvals(self.M) >= 0).all():
        #     print('不正定')
        print(self.get_error(self.M))
            
if __name__ == '__main__':
    a = SGD()
    a.solve()
    a.test()
    