# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:51:26 2018

@author: gaurav
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

trainDir = "/Users/gaura/Downloads/Machine_Learning/Assignment_1/datasets/regressionData/"
#Path where my training and test data exists

def loadDataset(fileName):
#This function will read the given csv file and return the data as a list.
    with open(fileName,"r") as csvfile:# Opening the file in read mode
        lines = csv.reader(csvfile)
        data = list(lines)
        return data# Returning the data as a list

def euclideanDistance(trainData, testData):
#This function calculates Euclidean Distance based on the below formula:
#
#      distance = ((Ai - Bi)**2)*(1/2)
#       where A and B are 2D and 1D numpy arrays respectively. 
#
    dists = np.sqrt(np.sum((trainData - testData)**2, axis=1))# Calculating Euclidean Distance
    sIArray = np.argsort(dists)# Getting the indices of sorted distances array
    return dists, sIArray# Returning the distances and sorted indices arraysy

def getRegressionFromNeighbours(trainData, testData, k):
    
    neighbours = []# list that will contain k nearest neighbours
    
    slicedTestData = testData[:-1]# Slicing only the feature data i.e. first 12 columns in this case
    slicedTrainData = trainData[:, :-1]# Slicing only the feature data i.e. first 12 columns in this case
    
    dist, indices = euclideanDistance(slicedTrainData, slicedTestData)# Calling Euclidean Distance Function
        
    for j in range(k):
        neighbours.append(trainData[indices[j]])# Appending k nearest neighbours in neighbours list
    
    neighbours = np.array(neighbours)# Converting list to numpy arra
    
    result = np.mean(neighbours[:,-1])# Calculating mean of the Class(Regression) values of k nearest neighbours
    
    return result# Returning the result

def getRSquaredValue(testData, predictions):
# This function calculates the R^2 value for the model   
    TSS = 0# Initializing Total Sum of Squares to 0
    SSR = 0# Initializing Sum of Squared Residuals to 0
    
    mean = 0
    mean = np.mean(testData[:,-1])# Calculating mean of the Class(Regression) values of test dataset
    for i in range(len(testData)):
        TSS += (testData[i,-1] - mean)**2
        # Incrementing Total Sum of Squares as per the below formula:
        #                   _
        #      TSS += (Yi - Y)**2
        #
        SSR += (predictions[i] - testData[i,-1])**2
        # Incrementing Sum of Squared Residuals as per the below formula:
        #
        #       SSR += (F(Xi) - Yi)**2
        #
    
    R2 = (1 - (SSR/TSS))# Calculating R^2

    return round(R2,4)# Returning value of R^2
    
def main():
# This is the main function
    
    trainData = loadDataset(trainDir + "trainingData.csv")# Calling the loadDataset Function passing the filename
    trainData = np.array(trainData, dtype=float)# Converting list to a nump array
    
    testData = loadDataset(trainDir + "testData.csv")# Calling the loadDataset Function passing the filename
    testData = np.array(testData, dtype=float)# Converting list to a nump array
    
    #k = 3# Factor specifying number of nearest neighbours to find
    kValue = []# list to store values of k
    accValue = []# list to store values of accuracy with respect to k
    
    for k in range(1, 20):
        kValue.append(k)# Appending value of k in kValue list
    
        predictions = []# List that will hold the predictions of the mo
    
        for x in range(len(testData)):
            result = getRegressionFromNeighbours(trainData, testData[x], k)
            #Calling getRegressionFromNeighbours function to calculate the Class(Regression) value for each test instance
            
            predictions.append(result)# Appending the result in predictions list
            print('> predicted=' + repr(result) + ', actual=' + repr(testData[x][-1]))
            #Printing the predicted Class(Regression) value and the actual Class(Regression) for each test instance
        
        R2 = getRSquaredValue(testData, predictions)
        # Calling getRSquaredValue function to get the accuracy of model
        print('R^2: ' + repr(R2))
        #Printing the accuracy of the model
        accValue.append(R2)# Appending value of accuracy in accValue list
        
    maxAccuracy = max(accValue)# Getting maximum value of accuracy
    valueOfk =  kValue[accValue.index(maxAccuracy)]# Value of k where the accuracy is maximum
    label1 = "Maximum Accuracy:",maxAccuracy, "at k:", valueOfk# Lable for the graph
    plt.plot(kValue, accValue, label=label1)# Plotting values of k and accuracy
    plt.xlabel('Number of Nearest Neighbours')# Setting x-axis label of graph
    plt.ylabel('R^2')# Setting y-axis label of graph
    plt.title('k-NN Regression Graph')# Setting title of graph
    plt.legend(loc='lower right')#Display area of label of graph
    plt.show#Printing the graph 
    
if __name__ == "__main__":
    main()# Calling the main function