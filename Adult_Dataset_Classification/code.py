# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:44:47 2018

@author: gaurav
"""

# -*- coding: utf-8 -*-

""" Importing the required libraries """
import warnings
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn import impute
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

""" Directory of the dataset """
datasetDir = "/Users/gaura/Downloads/Machine_Learning/Assignment_2/adult/"

""" Function to load the dataset in a DataFrame """
def loadDataset():
    
    adult_train = pd.read_csv(datasetDir + "adult.csv", na_values = '?', delimiter=",")# Reading the dataset
    
    labelCount = adult_train["income"].value_counts()# Counting record count of distinct labels
    print(labelCount)# Printing the label count of each label
    
    print ("Minority class is ",(labelCount[1]/len(adult_train))*100, " % of the dataset")# Printing percentage of minority class in the dataset
    
    adult_train["income"] = adult_train["income"].map({ "<=50K": -1, ">50K": 1 })# Mapping the string labels to integers
    
    adult_features = adult_train[adult_train.columns.difference(['income'])]# Extracting only Feature Data
    adult_class = adult_train["income"]# Extracting label data
    
    return adult_features, adult_class# Returning feature and label data
    
""" Function to perform Pre-processing """
def performPreprocessing(dataset):
    
    impDict = dataset.isna().sum()# Getting features and their respective number of missing values
    non0Fea = [name for name, count in impDict.items() if count != 0]# Extracting features with missing values
    
    imp_frequent = impute.SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')# Simple Imputer object for most_frequent strategy
    imp_mean = impute.SimpleImputer(missing_values=np.nan, strategy = 'mean')# Simple Imputer object for mean strategy
    
    #encoder = LabelEncoder()# Object for OrdinalEncoder
    encoder = OneHotEncoder(sparse=False)
    
    for feature in dataset.columns:# Looping for all the features in the dataset
    
        if dataset.dtypes[feature] == np.object:# Checking if the feature is of type Object

            if feature in non0Fea:# Checking if the feature has missing values
                dataset[[feature]] = imp_frequent.fit_transform(dataset[[feature]])# Imputing the missing values by most_frequent strategy

            # Encoding nominal features using label encoder
            #dataset[[feature]] = encoder.fit_transform(dataset[[feature]])# Encoding the categorical feature
        
        else:# If the feature if of type int64
            
            if feature in non0Fea:# Checking if the feature has missing values
                dataset[[feature]] = imp_mean.fit_transform(dataset[[feature]])# Imputing the missing values by mean strategy
    
    #Encoding nominal features using one-hot encoder
    dataset = encoder.fit_transform(dataset[["workclass", "education",	"marital.status",	"occupation",	"relationship",	"race"	,"sex",	"native.country"]])
    
    scaler = StandardScaler()# Creating the object of StandardScaler for scaling the dataset
    
    scaled_data = scaler.fit_transform(dataset)# Calling the fit_transform function of StandardScaler object to scale the dataset
    
    dataset = pd.DataFrame(scaled_data)# Converting the scaled data to a data frame
    
    return dataset# Returning the dataset as a dataframe

""" Function for selecting best discriminative features """    
def featureSelection(train_f, train_c):
    
    dtc = DecisionTreeClassifier(random_state = 40)# Creating object of DecisionTreeClassifier model for feature selection
    
    selector = RFECV(estimator = dtc, step=1, cv=10, scoring = 'accuracy')# Recursively eliminating features for finding best features
    
    selector.fit(train_f, train_c)# Training the RFECV
    
    rank = selector.ranking_# Get ranking of features
    print(rank)# Printing ranks
    index = [i for i in range(len(rank)) if rank[i] != 1]# Selecting weak features for droping
    
    names = train_f.columns[index]# Getting names of weak features
    print(names)# Printing names of weak features
    
    train_f.drop(names, axis = 1, inplace = True)# Dropping weak featured from the dataset
    
    return train_f# Returning the updated feature data
    
""" Function to get the best model """
def bestModel(train_f, train_c):
    
    models = []# List to contain models
    #Appending the models list with Various Classification models
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RFC', RandomForestClassifier(random_state=40)))
    models.append(('DTC', DecisionTreeClassifier(random_state=30)))
    models.append(('GNB', GaussianNB()))
    models.append(('SVM', SVC(random_state=32)))
    
    accuracy = []# List to store f1 scores
    names = []# List to store names of models
    
    for name, model in models:# Looping for all the models in models list
        StKfold = StratifiedKFold(n_splits=10, random_state=80)# Performing stratified cross fold
        cv_results = cross_val_score(model, train_f, train_c, cv=StKfold, scoring='f1')# Getting cross val score of each model
        accuracy.append(cv_results)# Appending the accuracy list with f1 score of each model
        names.append(name)# Appending the names list with the name of each model
        print("Model : {} | Accuracy : {} | Std.Dev. : {}".format(name,cv_results.mean(),cv_results.std()))
        # Printing the accuracy and std. deviation of each model

""" Function to perform over-sampling """
def oversampling(train_f, train_c, x):
    
    t_f = train_f.copy()# Copying feature data
    t_c = train_c.copy()# Copying label data
    
    if x==0:# Perform SMOTE oversampling
        print("SMOTE")
        sm= SMOTE(random_state=12)
        t_f, t_c= sm.fit_sample(t_f, t_c)
        
    if x==1:# Perform BorderlineSMOTE oversampling
        print("BorderlineSMOTE")
        sm= BorderlineSMOTE(random_state=14)
        t_f, t_c= sm.fit_sample(t_f, t_c)
        
    if x==2:# Perform BorderlineSMOTE oversampling
        print("SVM-SMOTE")
        sm= SVMSMOTE(random_state=16)
        t_f, t_c= sm.fit_sample(t_f, t_c)
        
    if x==3:# Perform BorderlineSMOTE oversampling
        print("ADASYN")
        sm= ADASYN(random_state=20)
        t_f, t_c= sm.fit_sample(t_f, t_c)
    
    return t_f, t_c# Returning oversampled dataset
        
""" Main Function """
if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')# Ignoring future warnings
    
    train_f, train_c = loadDataset()# Loading the datatset
    
    train_f = performPreprocessing(train_f)# Performing pre-processing on the dataset
    
    train_f = featureSelection(train_f, train_c)# Performing feature selection on the dataset
    

    for x in range(0, 4):# Looping 4 times
    
        t_f = train_f.copy()# Copying feature data
        t_c = train_c.copy()# Copying label data
    
        t_f, t_c = oversampling(t_f,t_c,x)# Performing oversampling
        
        
        ATrain_f, ATest_f, ATrain_c, ATest_c = train_test_split(t_f, t_c, test_size=0.33, random_state=40)
        # Splitting the dataset
        
        best = bestModel(train_f, train_c)# Get the best model for classification
        print(best)# Printing the Best model

    print("Hyper Parameter Optimization")
    
    # Parameter grid for Random Forest Classifier model
    param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split': [2, 6, 8, 10],
    'n_estimators': [100, 200, 300, 400],
    'criterion':['gini','entropy'],
    'random_state':[40]
    }
    
    clf= GridSearchCV(RandomForestClassifier(), param_grid, cv=10)# Performing hyper-parameter optimization using GridSearchCV on Random Forest model

    clf.fit(train_f, train_c)# Training the model
    
    print("\n Best parameters set found on development set:")
    print(clf.best_params_ , "with a score of ", clf.best_score_)# Printing the best parameters and best score
    scores = cross_val_score(clf.best_estimator_, train_f, train_c, cv=10)# Cross validation score for each combination of parameters
    print (scores.mean())# Printing the mean of scores