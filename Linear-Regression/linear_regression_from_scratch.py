#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kshitiz Bhandari

Implementation of simple linear regression from scratch to predict
total payment for all the claims in thousands of Swedish Kronor

Dataset used: Auto Insurance in Sweden
Downloaded from: data.world
In the dataset:
    x - number of claims (independent variable)
    y - total payment for all the claims in thousands of Swedist Kronor
"""
from math import sqrt
from random import randrange
from random import seed
from csv import reader

# average of values
def mean(x):
    '''
    x: list of numbers
    Returns the mean value of the list
    '''
    # return sum/n
    return sum(x) / float(len(x))


# sample covariance of two random variables
def covariance(x, y):
    '''
    Assumes x and y are list of numbers (two random variables)
    - need to be of the same length

    Returns sample covariance of x and y
    '''
    # assert that X and Y can take values (x_i, y_i)
    assert len(x) == len(y), "Sample size for two random variables do not match!"
    # initialize
    covar = 0.0
    # find averages of x and y
    mean_x = mean(x)
    mean_y = mean(y)
    
    # sample size
    n = len(x)
    # iterate through each item of the list
    for i in range(n):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    
    # return the covariance for the sample
    # For population, covariance = covar/n
    return covar/float(n - 1)


# sample variance of a variable
def variance(x):
    '''
    x: list of numbers (sample data)
    Returns: the sample variance of x
    '''
    # initialize
    var = 0.0
    # find mean
    mean_x = mean(x)
    # sample size
    n = len(x)
    for i in range(n):
        var += (x[i] - mean_x)**2
    # return sample variance
    return var/float(n - 1)
    
    
# root mean squared error
def rmse_error(y_actual, y_hat):
    assert len(y_actual) == len(y_hat), "The number of outcomes do not match!"
    # initialize
    sq_error = 0.0
    # number of outcomes
    n = len(y_hat)
    for i in range(n):
        sq_error += pow(y_hat[i] - y_actual[i], 2)
    
    # mean squared error
    mean_error = sq_error/float(n)
    # root mean squared error
    return sqrt(mean_error)


# calculate the coefficients for simple linear regression
def coefficients(dataset):
    '''
    Assumes dataset is a list of lists containing [x, y] pairs
    Returns the coefficients for simple linear regression
    '''
    # extract x and y from the dataset
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    
    # Calculate the coeffieicnts
    b1 = covariance(x, y) / variance(x)
    b0 = mean(y) - b1 * mean(x)
    
    return[b0, b1]


def linear_regression(train_set, test_set):
    '''
    Assumes train_set and test_set are list of lists containing [x, y] pairs
    Calculates the coefficients of simple linear regression using the training
        set
    Returns the prediction using the same coefficients on the test_set
    '''
    # initialize
    predictions = list()
    # evaluate coeffieicnts using the training dataset
    b0, b1 = coefficients(train_set)
    
    for row in test_set:
        #LR Model: y_hat = b0 + b1*x
        y_hat = b0 + b1 * row[0]
        # add prediction
        predictions.append(y_hat)
    
    return predictions


# function to split a dataset into training and test set
def train_test_split(dataset, split_size = 0.8):
    '''
    dataset : list of lists [x, y] pairs
    split_size : percentage of the dataset to be selected as the training set
        The default is 0.8.

    Returns training and test sets separated based on split_size
    '''
    # initialize
    train_set = list()
    # evaluate the length of training set
    train_set_size = split_size * len(dataset)
    
    # set the initial test set as a copy of the whole dataset
    # we'll remove elements as we populate the train_set
    test_set = list(dataset)
    
    while len(train_set) < train_set_size:
        # find a random index from 0 to length of the test set non-inclusive
        index = randrange(len(test_set))
        # remove the randomly selected row from the test set and
        # update the row into the training set
        train_set.append(test_set.pop(index))
    
    # return the two separated sets
    return train_set, test_set


# program to split a dataset and then use the training set to predict on the 
# test set and return the rmse using actual and predicted outcomes
def evaluate_model_rmse(dataset, algorithm, split_size = 0.8):
    '''
    dataset: list of lists [x, y] pairs
    algorithm: function that returns predictions using a ML model
    split_size: percentage of the dataset to be selected as the training set
        The default is 0.8.
    
    Using user-defined functions,
        - Splits the given dataset into training and test sets 
        - evaluates the prediction using the algorithm on the test_set
        - calculates the rmse
    Returns: rmse
    '''
    # split the dataset into training and test sets
    train_set, test_set = train_test_split(dataset, split_size)
    
    # evaluate the predictions using the given algorithm and the split datasets
    y_hat = algorithm(train_set, test_set)
    
    # extract actual y values from the test_set
    y_actual = [row[-1] for row in test_set]
    # calculate rmse using the values
    rmse = rmse_error(y_actual, y_hat)
    # return the calculated root mean squared error
    return rmse


 # To load a CSV file
def load_csv_file(filename):
    '''
    Loads a csv file with the given filename
    Returns a list of rows containing [x, y] (lists inside a list)
    '''
    dataset = list()
    # open file for extracting dataset and close automatically
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        line_count = 0
        # iterate through the rows
        for row in csv_reader:
            # do not store the header row
            if line_count == 0:
                line_count += 1
                continue
            else:
                # store other rows
                line_count += 1
                dataset.append(row)
    # return the rows - except the header - as lists inside a list
    return dataset


def str_column_to_float(dataset, column):
    '''
    Inputs a dataset and columns
    Changes the columns from string to float
    '''
    for row in dataset:
        row[column] = float(row[column].strip())


seed(1)
# specify file
filename = 'insurance.csv'
# load file and extract dataset
dataset = load_csv_file(filename)

# convert both x and y columns to float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

# specify split size
split_size = 0.8
# evaluate the algorithm
rmse = evaluate_model_rmse(dataset, linear_regression, split_size)
# Express results
print('Using Simple Linear Regression with training set as %.2f of the dataset' % (split_size))
print('The RMSE was calculated to be:')
print('\tRMSE = %.3f thousands of Swedish Kronor' %(rmse))



    