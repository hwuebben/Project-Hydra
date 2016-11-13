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


def quadratic(x, a, b, c):
    if a+b*x[0]+c*x[0]**2 < x[1]:
        return -1
    else:
        return 1


class Perceptron:

    def __init__(self, dim, target):
        self.w = np.random.rand(dim)
        self.target = target
        self.classValue = 0
        self.learns = 0

        #lernrate:
        self.nu = 0.05

    def newIteration(self):
        print("Adjusted weightvector: %d." % self.learns)
        self.learns = 0

    def classify(self, x):
        self.classValue = np.dot(self.w, x)
        if self.classValue > 0:
            return self.target
        return -1

    def getClassValue(self):
        return self.classValue
    
    # Perform a learning step for a given training datum with input values x
    # and output value y in {-1,1}
    # @param x The given input instance
    # @param y The desired output value
    # @return False if the perceptron did not produce the desired output value, i.e. the learning adaptation has been performed
    #         True if the perceptron already produced the correct output value, i.e. no adaptation has been performed
    def learn(self, x, y):

        yh = self.classify(x)
        #print("Target: ", self.target)
        #print(x)
        #print("Hypoth: ", yh)
        #print("Real:   ", y)
        #print("ClassV: ", self.classValue)
        if int(y) != int(yh):
            #print("Learn: ", np.multiply(self.nu*(1-self.classValue), x))
            self.w += np.multiply(self.nu*(1-self.classValue), x)
            self.learns += 1
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
        self.pocketError = np.inf
        while(not done and iterations < maxIterations):
            done = True
            iterations += 1
            it = iterator(fileName)
            for el in it:
                if int(el[1]) != self.target:
                    y = -1
                else:
                    y = 1
                noAdapt = self.learn(phi(el[0]), y)
                done = done and noAdapt
                if(not noAdapt):
                    cnt += 1

            currentError,currentErrorRel = self.calcError(iterator, fileName, phi)
            #print(self.w," ",currentError," ",cnt)
            if currentError < self.pocketError:
                #print("found better weights: ",self.w)
                self.putPocket(currentError,currentErrorRel)
        print("bester relativer P",self.target,": ",self.pocketErrorRel)
        print("bestes w P",self.target,": ",self.pocketW)
        #nach dem lernen: setze w auf das pocketW:
        self.getPocket()
        return cnt
    def putPocket(self,pocketError,pocketErrorRel):
        self.pocketW = self.w
        self.pocketError = pocketError
        self.pocketErrorRel = pocketErrorRel
    def getPocket(self):
        self.w = self.pocketW

    def calcError(self, iterator, fileName, phi):
        iterator = iterator(fileName)
        error = 0
        cnt = 0
        for x, y in iterator:
            _, yh = self.classify(phi(x))
            yh = int(yh)
            y = int(y)
            if y != self.target:
                y = -1
            else:
                y = 1

            if y != yh:
                error += 1
            cnt += 1
            #falls der Fehler schon groeßer ist: aufhoeren!
            if error > self.pocketError:
                break
        errorRel = error / cnt
        return [error,errorRel]

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