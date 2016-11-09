# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:27:20 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
import csv
from perceptron import Perceptron
from featureExtraction import *


"""
Iterator that yields all training data line-wise
@param fileName The name of the file that is to be read
@return Next line of the specified file as picture and class information
"""
def getNextPic(fileName):
    # Get the total number of lines in the given file
    with open(fileName) as f:
        numLines = sum(1 for _ in f)
    # Iterate over every line (sample)
    with open(fileName) as f:
        # Read comma-seperated-values
        content = csv.reader(f)
        # Iterate over every sample
        for idx,line in enumerate(content):
            # Terminate when eof reached
            if(idx == numLines):
                break
            # yield sample-image as 28x28 pic and the associated class
            yield np.reshape(line[1:], [28,28]).astype(int), int(line[0])

"""
Dummy transformation function (phi(x)) that just calculates the sum of all pixels
@param rawData The input picture
@return The input vector in the phi-space
"""
def transform(rawData):
    return [featureExtraction.calcVarX(rawData),featureExtraction.calcVarY(rawData)]

"""
Calculate the error on a dataset as percentage wrong classified
@param fileName The file containing the data
@param perceptron The perceptron that is to be evaluated
@param phi The transformation function (phi(x))
@return The error percentage
"""
def calculateError(fileName, perceptron, phi, target):
    iterator = getNextPic(fileName)
    error = 0
    cnt = 0
    for x, y in iterator:
        _, yh = perceptron.classify(phi(x))
        yh = int(yh)
        y = int(y)
        if y != target:
            y = -1
        else:
            y = 1

        print(y, yh)
        if y != yh:
            error += 1
        cnt += 1
    return error/cnt

if __name__ == "__main__":
    #auf welche Zahl soll trainiert werden?
    target = 0
    #initialiseren das Perceptron mit der Anzahl der features und der Zahl auf die traniert werden soll
    p = Perceptron(2,target)
    phi = transform
    fileName = "mnist_first_batch.csv"
    iterator = getNextPic(fileName)
    p.learnIteratorDataset(getNextPic, fileName, transform, maxIterations=10)
    print(calculateError(fileName, p, phi,target)*100,'%')
    
    