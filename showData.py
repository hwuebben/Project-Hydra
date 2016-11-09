# -*- coding: utf-8 -*-
"""
Simple program to show how to read and use the MNIST dataset
The file mnist_first_batch.csv must be in the path when the program is run

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
import csv
from matplotlib import pyplot as plt


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
Plot a specified number of pictures from the mnist dataset
@param fileName The name of the file that is to be read
@param rows The desired number of rows
@param cols The desired number of columns
@param offset (optional) Skips the first offset lines
"""
def plotDigits(fileName, rows, cols, offset=0):
    # Create new iterator for iterating the file
    iterator = getNextPic(fileName)
    # Skip the desired offset in file
    for i in range(offset):
        pic,_ = iterator.next()
    fig, axarr = plt.subplots(rows, cols)
    for row in axarr:
        for i in range(cols):
            # Get the next picture, discard the class information
            pic,_ = next(iterator)
            # Plot the picture
            plt.xlim([0,27])
            plt.ylim([0,27])
            row[i].pcolor(pic[::-1], cmap='Greys')
    plt.show()

"""
Count and print the number of instances for each class
@param fileName The name of the file that is to be read
"""
def countSample(fileName):
    # Create bins for counting
    count = np.zeros(10)
    # Open the file
    with open(fileName) as f:
        # Read the file as a csv-file
        content = csv.reader(f)
        # Count the occurances of each class
        for line in content:
            count[int(line[0])] += 1
    # Print the findings
    for i in range(10):
        print(int(count[i]),' instances for class ', i)
            

if __name__ == "__main__":
    countSample("mnist_first_batch.csv")
    plotDigits("mnist_first_batch.csv", 5, 4)
    
    