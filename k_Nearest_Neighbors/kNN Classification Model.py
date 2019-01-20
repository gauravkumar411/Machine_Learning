# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:03:24 2018

@author: gaurav
"""
import numpy as np
import csv
import operator
import matplotlib.pyplot as plt

datasetDir = ""
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

def getNeighbours(trainData, testData, k):
# This function finds the k nearest neighbours for the given test instance
    
    neighbours = []# list that will contain k nearest neighbours

    slicedTestData = testData[:-1]# Slicing only the feature data i.e. first 5 columns in this case
    slicedTrainData = trainData[:, :-1]# Slicing only the feature data i.e. first 5 columns in this case
    
    dist, indices = euclideanDistance(slicedTrainData, slicedTestData)# Calling Euclidean Distance Function
        
    for j in range(k):
        neighbours.append(trainData[indices[j]])# Appending k nearest neighbours to the neighbours list
            
    return neighbours# Returning list of neighbours

def getVotes(neighbours):
# This function calculates votes for the k nearest neighbours     
    votes = {}# Dictionary that will contain all the Classes their Votes respectively.
    for i in range(len(neighbours)):
        Class = neighbours[i][-1]# Getting the Class of neighbour
        if Class in votes:# Condition to check if the Class already exists in dictionary
            votes[Class] += 1# Incrementing the vote by 1 for that particular class
        else:
            votes[Class] = 1# Inserting the Class in dictionary
    maxVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse = True)
    # Sorting the dictionary in decreasing order with respect to the 2nd column i.e. votes
    
    return maxVotes[0][0]# Returning the Class of the maximum number of votes

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
    
        for x in range(len(normalizedTestData)):
            neighbours = getNeighbours(normalizedTrainData, normalizedTestData[x], k)
            #Calling getNeighbours function to find k nearest neighbours for each test instance
        
            result = getVotes(neighbours)
            #Calling getVotes function to get the Class with maximum number of votes
            
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
    plt.title('k-Nearest Neighbours Graph')# Setting title of graph
    plt.legend(loc='lower right')#Display area of label of graph
    plt.show#Printing the graph    
    
if __name__ == "__main__":
    main()# Calling the main function