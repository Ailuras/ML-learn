import numpy as np
import matplotlib.pyplot as plt
import itertools
from regex import W
from tqdm import tqdm
import random
import math

class SVRG:
    def __init__(self, dataFile='Data/covtype/covtype.libsvm.binary.scale', alpha=0.1, epsilon=0.0001, times=10000, features=54, m=1000000, bench_size=30):
        self.dataFile = dataFile
        self.alpha = alpha
        self.times = times
        self.epsilon = epsilon
        self.bench_size = bench_size
        self.features = features+1
        self.m = m
        self.value = []
        self.get_info(self.dataFile)
        self.omega = np.zeros((1 ,self.features))
        
    def get_info(self, dataFile):
        y_data = []
        x_data = []
        with open(dataFile) as f:
            line = f.readline()
            while line:
                y_data.append(int(line[0])-1)
                info = line[2:-2].split(' ')
                x = [0]*(self.features)
                x[0] = 1
                for i in info:
                    i = i.split(':')
                    x[int(i[0])] = float(i[1])
                x_data.append(x)
                line = f.readline()
            self.y_data = np.mat(y_data)
            self.x_data = np.mat(x_data)
            self.size = self.y_data.shape[1]
            print(self.y_data.shape)
            print(self.x_data.shape)
        
    def get_error(self):
        error = 0
        for i in tqdm(range(self.size)):
            error += self.y_data[0, i]*math.exp(self.get_h(i))+(1-self.y_data[0, i])*math.exp(1-self.get_h(i))
        return error/self.size
    
    def get_h(self, index):
        return 1/(1+math.e**((-self.omega*self.x_data[index].T)[0, 0]))
    
    def get_gradient(self, index):
        # using logistic regression loss
        return (self.get_h(index)-self.y_data[0, index])*self.x_data[index]
    
    def solve(self):
        d = np.zeros((1 ,self.features))
        gradient = np.zeros((self.size, self.features))
        for i in tqdm(range(self.times)):
            G = np.zeros((1 ,self.features))
            for i in range(self.size):
                gradient[i] += self.get_gradient(i)
                G += gradient[i]
            omega = self.omega
            for j in range(self.m):
                index = random.randint(0, self.size-1)
                g = self.get_gradient(index)
                d = G-gradient[index]+g
                omega -= self.alpha*d
            index = random.randint(0, self.size-1)
            self.value.append(self.y_data[0, index]*math.exp(self.get_h(index))+(1-self.y_data[0, index])*math.exp(1-self.get_h(index)))
            self.omega = omega
    def draw(self):
        x = range(self.times)
        y = self.value
        plt.plot(x, y, label="Train_Loss_list")
        plt.show()
        
    def test(self):
        d = np.zeros((1 ,self.features))
        gradient = np.zeros((self.size, self.features))
        g = self.get_gradient(0)
        print(g)
        d = d-gradient[0]+g
        print(d)
            
if __name__ == '__main__':
    a = SVRG()
    print('done!')
    print(a.get_error())
    a.solve()
    print(a.get_error())
    # a.test()