# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:27:20 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
import csv
from perceptron import Perceptron
from featureExtraction import FeatureExtraction
from perceptron import Perceptron
import numpy as np
import sys
import os
import pickle
import copy


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


def calculate_error(iterator, perceptrons, feature_extraction):
    """
    Calculate the error on a dataset as percentage wrong classified
    @param iterator The iterator containing the data
    @param perceptrons The perceptrons array
    @param feature_extraction The transformation class instance)
    @return array (containing error percentage for every perceptron), the error percentage for all perceptrons
    """
    error = 0
    cnt = 0
    errors = np.zeros(10)
    nrSamples = np.zeros(10)
    nrClass = np.zeros(10)
    errCnt = np.zeros(10)
    for x, y in iterator:
        nrSamples[y] += 1
        yh = classify(perceptrons, feature_extraction.get_train_data(x))
        if yh >= 0:
            nrClass[yh] += 1
        if int(y) != int(yh):
            errors[y]+= 1
            error += 1
        cnt += 1
        if cnt % 1000 == 0:
            print("Classified %d pictures" % cnt)
    print("Errors: ", errors)
    print("Sample: ", nrSamples)
    print(nrClass)
    for y,e in enumerate(errors):
        errCnt[y] = errors[y] / nrSamples[y]
        print("Fehlerquote fuer ",y,": ",errors[y]/nrSamples[y])

    return errCnt, error/cnt


def classify(perceptrons, x, y=None):
    max_class_value = -1
    classification = -1

    # Cheating Mode, only ask the correct Classifier
    if y is not None:
        return perceptrons[y].classify(x)

    for p in perceptrons:
        yh = p.classify(x)
        class_value = p.classValue
        if class_value >= max_class_value:
            max_class_value = class_value
            classification = yh
        #if(y == realy):
            #print("Wert fuer richtiges P: ",classValue)
    return classification


if __name__ == "__main__":

    USAGE = "python evaluateMNIST.py <learn> [pocket]| <classify path/to/mnist/test/batch.csv>\n" \
            "LEARN:    'mnist_first_batch.csv' has to be in the same directory\n" \
            "The weight-vector of the perceptrons will be dumped into wPs.pic " \
            "if pocket, uses pocket perceptron (longer computation time)\n\n" \
            "CLASSIFY: 'wps.pic' has to be in the same directory "

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit()


    wvFilename = 'wPs.pic'
    batchFilename = 'mnist_first_batch.csv'

    # specify amount of iterations over the whole dataset
    iterations = 100

    # initializes feature extraction instance
    fe = FeatureExtraction()
    perceptrons = []
    usePocket = False
    pocket = [None, None, None, None, None, None, None, None, None, None]
    pocketErr = np.ones(10)

    if len(sys.argv) == 3 and sys.argv[2] == "pocket":
        usePocket = True

    if sys.argv[1] == "classify":
        if not os.path.exists(sys.argv[2]):
            print("Error:", sys.argv[2], "does not exist!")
            sys.exit()
        batchFilename = sys.argv[2]

    iterator = getNextPic(batchFilename)

    # learn dataset
    if sys.argv[1] == "learn":

        # create perceptron for every target
        for target in range(10):
            # initialiseren das Perceptron mit der Anzahl der features und der Zahl auf die traniert werden soll
            perceptrons.append(Perceptron(fe.return_length, target))

        i = 0
        for iteration in range(iterations):
            # iterate over whole dataset
            for x, y in getNextPic(batchFilename):
                if i % 1000 == 0:
                    print("Processed %d pictures" % i)

                # do only on feature extraction and learn every perceptron simultaneously
                tx = fe.get_train_data(x)

                for p in perceptrons:
                    p.learn(tx, y)
                    #input("Press Enter to continue...")

                i += 1

            # calculate errors of every perceptron
            if usePocket:
                print("Calculate error for best pocket")
                err, _ = calculate_error(getNextPic(batchFilename), perceptrons, fe)

            # check if a perceptrons already finished learning
            for p in perceptrons:
                idx = int(p.target)
                # save better weightvectors
                if usePocket and pocketErr[idx] >= err[idx]:
                        print("Update pocket for target %d" % p.target)
                        pocket[idx] = copy.deepcopy(p)
                        pocketErr[idx] = err[idx]

                # remove finished perceptrons from iteration
                if not p.learns:
                    print("Perception with target %d finished learning: " % p.target)
                    pocket[idx] = p
                    perceptrons.remove(p)
                else:
                    p.newIteration()
            i = 0
            print("Finished iteration %d." % iteration)

        for p in perceptrons:
            pocket[int(p.target)] = p

        with open(wvFilename, "wb") as f:
            pickle.dump(pocket, f)

        print("Printing final stats, if you don't want to see then, just ctrl-C, weight vector is already saved")
        _, err = calculate_error(getNextPic(batchFilename), pocket, fe)
        print(err * 100, '%')

    if sys.argv[1] == "classify":
        with open(wvFilename, "rb") as f:
            perceptrons = pickle.load(f)

        _, err = calculate_error(getNextPic(batchFilename), perceptrons, fe)
        print(err * 100, '%')

