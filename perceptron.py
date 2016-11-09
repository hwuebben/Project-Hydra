# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:36:16 2016

@author: joschnei
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib
  

def func(x, a, b):
    if a+b*x[0] < x[1]:
        return -1
    else:
        return 1

def quadratic(x,a,b,c):
    if a+b*x[0]+c*x[0]**2 < x[1]:
        return -1
    else:
        return 1

class Perceptron:
    def __init__(self, dim,target):
        self.w = np.random.rand(dim+1)
        self.target = target
#        self.w = np.zeros(dim+1)
    def classify(self, x):
        x = np.insert(x, 0, 1)
        return x, np.sign(np.dot(self.w, x))
    
    # Perform a learning step for a given training datum with input values x
    # and output value y in {-1,1}
    # @param x The given input instance
    # @param y The desired output value
    # @return False if the perceptron did not produce the desired output value, i.e. the learning adaptation has been performed
    #         True if the perceptron already produced the correct output value, i.e. no adaptation has been performed
    def learn(self, x, y):
        x, yh = self.classify(x)
        if(int(y) != int(yh)):
            self.w += y*x
            return False
        return True

    # Perform the complete perceptron learning algorithm on the dataset (x_i, y_i)
    # @param dataset The complete dataset given as a 2D list [inputvalues, outputvalues]
    # with inputvalues being a list of all input values which again are a list of coordinates for each dimension
    # and output values a list of all desired output values
    def learnDataset(self, dataset):
        done = False
        cnt = 0
        while(not done):
            done = True
            for el in dataset:
                noAdapt = self.learn(el[0], el[1])
                done = done and noAdapt
                if(not noAdapt):
                    cnt += 1
        return cnt
    
    def learnIteratorDataset(self, iterator, fileName, phi, maxIterations=1):
        done = False
        cnt = 0
        iterations = 0
        while(not done and iterations < maxIterations):
            done = True
            iterations += 1
            it = iterator(fileName)
            for el in it:
                if not el[1] == self.target:
                    y = 0
                else:
                    y = 1

                noAdapt = self.learn(phi(el[0]), y)
                done = done and noAdapt
                if(not noAdapt):
                    cnt += 1
        return cnt

    def plot(self, pts=None, mini=-1, maxi=1, res=500):
        font = {'family' : 'serif',\
        'weight' : 'bold',\
        'size'   : 22}
        matplotlib.rc('font', **font)
        delta = (maxi - mini) / res
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        plt.xlim([mini, maxi])
        plt.ylim([mini, maxi])
        xcoord = np.linspace(mini, maxi, res, endpoint = True)
        ycoord = np.linspace(mini, maxi, res, endpoint = True)
        result = np.zeros([res, res])
        for i, x in enumerate(xcoord):
            for j, y in enumerate(ycoord):
                _, result[i][j] = self.classify([x,y])
        plt.pcolor(xcoord, ycoord, result, cmap='coolwarm')
        ptpx = []
        ptpy = []
        ptmx = []
        ptmy = []
        if pts != None:
            for pt in pts:
                if pt[1] < 0:
                    ptmx.append(pt[0][1])
                    ptmy.append(pt[0][0])
                else:
                    ptpx.append(pt[0][1])
                    ptpy.append(pt[0][0])
            plt.scatter(ptpx, ptpy, marker='x',color='k', s=50, linewidth=2)
            plt.scatter(ptmx, ptmy, marker='o',color='k', s=50, linewidth=2)

    def initWeights(self, dataX, dataY):
        on = np.array([np.ones(len(dataX))])
        dataX = np.concatenate((dataX,on.T),axis=1)
        x = np.matmul(np.linalg.inv(np.matmul(np.transpose(dataX), dataX)), np.transpose(dataX))
        self.w = np.matmul(x, dataY)

if __name__ == "__main__":
    p = Perceptron(2)
    np.random.seed(1234)
    dataX = np.random.rand(100,2)*2-1
    dataY = np.zeros(100)
    for i,x in enumerate(dataX): 
        dataY[i] = func(x, -0.7, 0.1)
#    p.initWeights(dataX, dataY)
    p.plot(zip(dataX,dataY))