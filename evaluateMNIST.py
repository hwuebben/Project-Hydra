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
    fe = featureExtraction(rawData)
    features = []
    features.extend(fe.calcVar())
    features.extend(fe.calcMinMax())
    features.extend(fe.normalizedDists())

    return features

"""
Calculate the error on a dataset as percentage wrong classified
@param fileName The file containing the data
@param perceptron The perceptron that is to be evaluated
@param phi The transformation function (phi(x))
@return The error percentage
"""
def calculateError(fileName, perceptrons, phi):
    iterator = getNextPic(fileName)
    error = 0
    cnt = 0
    errors = np.zeros(10)
    nrSamples = np.zeros(10)
    nrClass = np.zeros(10)
    for x, y in iterator:
        nrSamples[y] += 1
        yh = classify(perceptrons,phi,x,y)
        nrClass[yh] += 1
        if y != yh:
            errors[y]+= 1
            error += 1
        cnt += 1
    print(errors)
    print(nrSamples)
    print(nrClass)
    for y,e in enumerate(errors):
        print("Fehlerquote fuer ",y,": ",errors[y]/nrSamples[y])

    return error/cnt

def classify(perceptrons, phi, x,realy):
    maxClassValue = -1
    for y,perceptron in enumerate(perceptrons):
        _, yh = perceptron.classify(phi(x))
        classValue = perceptron.getClassValue()
        if classValue >= maxClassValue:
            maxClassValue = classValue
            classification = y
        #if(y == realy):
            #print("Wert fuer richtiges P: ",classValue)
    return classification







if __name__ == "__main__":
    phi = transform
    fileName = "mnist_first_batch.csv"
    iterator = getNextPic(fileName)
    perceptrons = []
    #dreckiger trick um die ANzahl der Dimensionen nicht manuell eingeben zu muessen:
    nrDimensions = len(transform(np.arange(25).reshape(5,5)))
    maxIterations = 10
    for target in range(10):
        #initialiseren das Perceptron mit der Anzahl der features und der Zahl auf die traniert werden soll
        exec("p"+str(target)+" = Perceptron(nrDimensions,"+str(target)+")")
        print ("trainiere naechstes Perzeptron")
        exec("p"+str(target)+".learnIteratorDataset(getNextPic, fileName, transform, maxIterations)")
        exec("perceptrons.append(p"+str(target)+")")
    print("training fertig")
    print(calculateError(fileName, perceptrons, phi)*100,'%')

    