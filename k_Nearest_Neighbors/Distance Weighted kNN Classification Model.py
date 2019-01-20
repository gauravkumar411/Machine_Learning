# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:03:24 2018

@author: gaurav
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

trainDir = ""
#Path where my training and test data exists

def loadDataset(fileName):
#This function will read the given csv file and return the data as a list.
    with open(fileName,"r") as csvfile:# Opening the file in read mode
        lines = csv.reader(csvfile)
        data = list(lines)
        return data# Returning the data as a list

def normalize(data):
#This function normalizes the given data as per the below formula:
#       
#      f(x) = Xi - min
#            ----------
#             max - min

    for x in range(data.shape[1]-1):
        mini = np.min(data[:,x], axis = 0)# Taking minimum value of a particular column
        maxi = np.max(data[:,x], axis = 0)# Taking maximun value of a particular column
        for y in range(len(data)):
            data[y,x] = (data[y,x] - mini) / (maxi - mini)# Calculating normalized value
    return data# Returning normalized array of data

def euclideanDistance(trainData, testData):
#This function calculates Euclidean Distance based on the below formula:
#
#      distance = ((Ai - Bi)**2)*(1/2)
#       where A and B are 2D and 1D numpy arrays respectively. 
#
    dists = np.sqrt(np.sum((trainData - testData)**2, axis=1))# Calculating Euclidean Distance
    sIArray = np.argsort(dists)# Getting the indices of sorted distances array
    return dists, sIArray# Returning the distances and sorted indices arrays

def manhattanDistance(trainData, testData):
#This function calculates Manhattan Distance based on the below formula:
#
#       distance = |Ai - Bi|
#       where A and B are 2D and 1D numpy arrays respectively. 
#    
    dists = np.sum(abs(trainData - testData), axis=1)# Calculating Manhattan Distance
    sIArray = np.argsort(dists)# Getting the indices of sorted distances array
    return dists, sIArray# Returning the distances and sorted indices arrays

def minkowskiDistance(trainData, testData, n):
#
#       distance = ((Ai - Bi)**n)*(1/n)
#       where A and B are 2D and 1D numpy arrays respectively and n is the factorization value.
#    
    dists = (np.sum(abs((trainData - testData)**n), axis=1))**(1/n)# Calculating Minkowski Distance
    sIArray = np.argsort(dists)# Getting the indices of sorted distances array
    return dists, sIArray# Returning the distances and sorted indices arrays

def getDistanceWeightedVotes(trainData, testData, k, n, distanceMetric):
# This function finds k number of nearest neighbours for each class 
# and calculates votes using the inverse distance formula as below:  
#
#                  1
#       vote = ----------
#               d(Xq, Xi)
#
    slicedTestData = testData[:-1]# Slicing only the feature data i.e. first 5 columns in this case
    slicedTrainData = trainData[:, :-1]# Slicing only the feature data i.e. first 5 columns in this case
    
    if distanceMetric == 1:
        dist, indices = euclideanDistance(slicedTrainData, slicedTestData)# Calling Euclidean Distance Function
    
    if distanceMetric == 2:
        dist, indices = manhattanDistance(slicedTrainData, slicedTestData)# Calling Manhattan Distance Function
        
    if distanceMetric == 3:
        dist, indices = minkowskiDistance(slicedTrainData, slicedTestData, n)# Calling Manhattan Distance Function
    
    voteForClass0 = 0# Initializing votes counting variable for Class 0.0
    voteForClass1 = 0# Initializing votes counting variable for Class 1.0

    neighClass0 = []#List that will contain k nearest neighbours for Class 0.0
    
    j = -1
    while len(neighClass0) <= k-1:#Loop to find k nearest neighbours for Class 0.0
        j += 1
        if trainData[indices[j], -1] == 0.0:
            neighClass0.append(trainData[indices[j]])#Appending neighbour to the list
            voteForClass0 += 1/(dist[indices[j]])#Calculating total number of votes for Class 0.0
    
    neighClass1 = []#List that will contain k nearest neighbours for Class 1.0
    i = -1
    while len(neighClass1) < k-1:#Loop to find k nearest neighbours for Class 1.0
        i += 1
        if trainData[indices[i], -1] == 1.0:
            neighClass1.append(trainData[indices[i]])#Appending neighbour to the list
            voteForClass1 += 1/(dist[indices[i]])#Calculating total number of votes for Class 0.0
    
    dict = {voteForClass0 : 0.0, voteForClass1 : 1.0}# Dictionary of number of votes and their respective Class     

    return dict[max(voteForClass0,voteForClass1)]# Returning the Class having maximum number of votes
    
def getAccuracy(testData, predictions):
# This function is calculating the accuracy of the model.	
    correct = 0# Counter to counter the number of correct predictions
    
    for i in range(len(testData)):
        
        if testData[i][-1] == predictions[i]:#Condition to check if the prediction is correct
            correct += 1# Incrementing the counter
            
    return (correct/len(testData)) * 100# Returning the overall accuracy
    
def main():
# This is the main function
    
    trainData = loadDataset(trainDir + "trainingData2.csv")# Calling the loadDataset Function passing the filename
    trainData = np.array(trainData, dtype=float)# Converting list to a nump array
    normalizedTrainData = normalize(trainData)# Calling the normalization function to scale the data
    
    testData = loadDataset(trainDir + "testData2.csv")# Calling the loadDataset Function passing the filename
    testData = np.array(testData, dtype=float)# Converting list to a nump array
    normalizedTestData = normalize(testData)# Calling the normalization function to scale the data
    
    kValue = []# list to store values of k
    accValue = []# list to store values of accuracy with respect to k
    
    for k in range(1, 50):
        kValue.append(k)# Appending value of k in kValue list
    
        predictions = []# List that will hold the predictions of the model
    
        #k = 16# Factor specifying number of nearest neighbours to find
        
        n = 3# Factor Value for Minkowski Distacne Metric
        
        distanceMetric = 1
        #Distance Metric to use:
        # 1 --> Euclidean Distance
        # 2 --> Manhattan DIstance
        # 3 --> Minkowski Distance
    
        for x in range(len(normalizedTestData)):
        
            result = getDistanceWeightedVotes(normalizedTrainData, normalizedTestData[x], k, n, distanceMetric)
            #Calling getDistanceWeightedVotes function to get the prediction of Class for each test instance
        
            predictions.append(result)# Appending the result(Class) in predictions list
            print('> predicted=' + repr(result) + ', actual=' + repr(normalizedTestData[x][-1]))
            # Printig the predicted and actual value of Class for each test instance
        
        accuracy = getAccuracy(normalizedTestData, predictions)
        #Calling the getAccuracy function to fetch the overall accuracy of the model
    
        print('Accuracy: ' + repr(accuracy) + '%')# Printing the accuracy of the model.
        accValue.append(accuracy)# Appending value of accuracy in accValue list
        
    maxAccuracy = max(accValue)# Getting maximum value of accuracy
    valueOfk =  kValue[accValue.index(maxAccuracy)]# Value of k where the accuracy is maximum
    label1 = "Maximum Accuracy:",maxAccuracy, "at k:", valueOfk# Lable for the graph
    plt.plot(kValue, accValue, label=label1)# Plotting values of k and accuracy
    plt.xlabel('Number of Nearest Neighbours')# Setting x-axis label of graph
    plt.ylabel('Accuracy')# Setting y-axis label of graph
    plt.title('Distacne Weighted k-NN Graph using Euclidean Distance Metric')# Setting title of graph
    plt.legend(loc='lower right')#Display area of label of graph
    plt.show#Printing the graph     
        
if __name__ == "__main__":
    main()# Calling the main function