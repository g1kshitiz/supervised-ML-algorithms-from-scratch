#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kshitiz Bhandari

Implementation of Naive Bayes from scratch for a classification problem
Dataset used: Iris flower (downloaded from: kaggle.com)
"""

from math import sqrt, pi, exp
from csv import reader
## Summary statistics: mean and standard deviation

# average (mean) of values
def mean(x):
    '''
    x: list of numbers
    Returns the mean value of the list
    '''
    # return sum/n
    return sum(x) / float(len(x))


# sample standard deviation of a variable
def sd(x):
    '''
    x: list of numbers (sample data)
    Returns: the sample standard deviation of x
    '''
    # initialize
    variance = 0.0
    # find mean
    mean_x = mean(x)
    # add the deviation from the mean squared for each number in x
    for num in x:
        variance += pow(num - mean_x, 2)
    
    # sample variance for a sample size of n:
    # var = [sum(x_i - mean_x)^2 / (n - 1)] 
    variance = variance/float(len(x) - 1)
    
    # Instead of the above 4 lines, we can simply evaluate the following:
    #mean_x = mean(x)
    #variance = sum([(num - mean_x)**2 for num in x]) / float(len(x) - 1)
    
    # return sample standard deviation (which is the square root of variance)
    return sqrt(variance)


# Summarize a dataset using it's mean, standard deviation, and count
# (later for a Gaussian PDF)
def summarize_dataset(dataset):
    '''
    dataset: list of lists

    Returns: summary_statistics - a list of tuples with mean, sd, and length of
        each column in the dataset
    '''
    
    # * operator separates the dataset (list of lists) into separate lists
    # and the zip iterates over each list (row)
    summary_statistics = []
    # iterate over all the columns except the one containing class
    for column in range(len(dataset[0]) - 1):
        # find the average, standard deviation, and length of each column (feature)
        avg = mean([row[column] for row in dataset])
        stdev = sd([row[column] for row in dataset])
        sample_size = len([row[column] for row in dataset])
        # append a tuple of average, standard deviation and no. of samples
        summary_statistics.append((avg, stdev, sample_size))
    
    ## Can do the following (just one line) if the class column is a number
    #summary_statistics = [(mean(column), sd(column), len(column)) for column in zip(*dataset)]   
    ## * operator separates the dataset (list of lists) into separate lists
    ## and the zip iterates over each list (row)
    ## and delete the summary statistic for the last column (class)
    #del(summary_statistics[-1])

    return summary_statistics


# Need probability of data by class
# So, we will separate the dataset by class
def separate_dataset_by_class(dataset):
    '''
    dataset: a list of lists
    Returns a dictionary with:
        key: class name
        value: and a list of lists (instances of that class in the dataset) 
    '''
    # initialize
    result = dict()
    # iterate through the rows
    for i in range(len(dataset)):
        # current row
        instance = dataset[i]
        # the class of the current row
        class_name = instance[-1]
        # if the class is not already stored in the dictionary
        if class_name not in result:
            # assign an empty list to the value of the class
            result[class_name] = list()
        # add the current row to the appropriate class in the dictionary
        result[class_name].append(instance)
    # final dictionary with separated classes with their respoective instances 
    return result


# finding the summary of each class
def summarize_by_class(dataset):
    '''
    dataset: a list of lists
    First, separates the dataset by class
    Then, calculates summary statistics for each column within each class
    
    Returns: dictionary with
        key: class-name 
        value: a list of tuples containing summary statistics of each column
            within that class
    '''
    # first separate the dataset by class using previously defined function
    separated = separate_dataset_by_class(dataset)
    
    # initialize an empty dictionary
    summary_statistics = dict()
    
    # iterate through both keys and values
    for class_name, instances in separated.items():
        # assign summary of each column (tuples with mean, sd, length)
        # as the values within each class
        summary_statistics[class_name] = summarize_dataset(instances)
    
    # return the dictionary with summaries of columns within each class
    return summary_statistics
    


## Assuming the values of each feature follow a Gaussian (normal) distribution
# To calculate, the PDF (probability Density function)
def Gaussian_pdf(x, mu, sigma):
    '''
    x: value taken by the random variable
    mu: mean of the random variable
    sigma: standard deviation of the random variable
    
    Returns: the value of the Gaussian PDF at x
    '''
    # term to denote the exponent
    exponent = -(1/2) * pow( (x - mu)/sigma, 2) 
    # final Gaussian PDF
    return (1 / (sigma * sqrt(2 * pi)) * exp(exponent))


# Need to calculate the probability of a given instance (row of features)
# belonging to a certain class
# We calculate the probability that a new row of features belongs to the first class,
# then to the second class, and so on.
# Instead of calculating the actual probability using Bayes' theorem, a simplified
# implementation is used (because this function is used only to make prediction)
# The denominator from the Bayes' theorem is removed (constant for all)
def calculate_probabilities_for_each_class(summary_statistics, row):
    total_rows = sum(summary_statistics[num][0][2] for num in summary_statistics)
    # initialize dictionary  
    probabilities = dict()
    
    for class_name, class_summary in summary_statistics.items():
        # {class (key): probability of the row falling into this class(value)}
        probabilities[class_name] = summary_statistics[class_name][0][2]/float(total_rows)
        
        for i in range(len(class_summary)):
            mean_class, sd_class, count = class_summary[i]
            # not the true probability, however, it is valid for prediction
            probabilities[class_name] *= Gaussian_pdf(row[i], mean_class, sd_class)
    # not the true probability
    return probabilities

# To load dataset from a csv file
def load_csv_file(filename):
    '''
    Loads a csv file with the given filename
    Returns a list of rows
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


def predict(summary_statistics, row):
    '''
    summary_statistics: dictionary with
        key: class-name 
        value: a list of tuples containing summary statistics of each column
            within that class
    row: set of features
    
    Returns the class with the highest probability for the given row (set of features)
    '''
    # calculate probabilities for each class given the current row
    # (dictionary)
    probabilities = calculate_probabilities_for_each_class(summary_statistics, row)
    
    # initialize
    best_class, best_probability = None, -1
    # iterate through the classes and their respective probabilities
    for class_name, probability in probabilities.items():
        
        if best_class is None or probability > best_probability:
            # assign first class if none is assigned yet
            # or if the probability of the current class is higher than the highest so far
            best_probability = probability
            
            best_class = class_name
    
    # return the prediction using simplified implementation of Naive Bayes
    return best_class
            
    
    
# load the iris dataset from the csv file
dataset = load_csv_file('iris.csv')

# convert the features from string (datatype when reading file) to float
for column in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, column)


## test instances for prediction
test_row = [5.7, 2.9, 4.2, 1.3]
#test_row = [4.5, 2.3, 1.3, 0.3]


## Uncomment this section for manual entry of each feature
# print('Enter features to predict the species:')
# f1 = float(input('Enter sepal length: '))
# f2 = float(input('Enter sepal width:' ))
# f3 = float(input('Enter petal length: '))
# f4 = float(input('Enter petal width: '))
# test_row = [f1, f2, f3, f4]

# find the summaries
summarize = summarize_by_class(dataset)

# predict the class given the dataset and the given test row
result = predict(summarize, test_row)


# In this case, the algorithm was used to predict the species of only one test
# instance. However, it can be implemented for a test dataset.
# In that case, the accuracy can be evaluated for the test dataset
# Moroever, cross validation can be done and the algorithm can be tuned
print(f'For the given features: {test_row}, \nThe predictionn is:', result)

