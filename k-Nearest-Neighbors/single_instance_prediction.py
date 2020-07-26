#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kshitiz Bhandari

Using knn from scratch to predict the species of a flower given its features
Dataset used: Iris flower (downloaded from: kaggle.com)
"""
from csv import reader
from knn_from_scratch import KNN

 # To load a CSV file
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


# load the iris dataset from the csv file
dataset = load_csv_file('iris.csv')

# convert the features from string to float
for column in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, column)


knn = KNN(k = 5)

## test instances for prediction
#test_row = [5.7, 2.9, 4.2, 1.3]
test_row = [4.5, 2.3, 1.3, 0.3]


## Uncomment this section for manual entry of each feature
# print('Enter features to predict the species:')
# f1 = float(input('Enter sepal length: '))
# f2 = float(input('Enter sepal width:' ))
# f3 = float(input('Enter petal length: '))
# f4 = float(input('Enter petal width: '))
# test_row = [f1, f2, f3, f4]


# predict the class (in this case the species of the flower)
result = knn.predict(dataset, test_row)

# In this case, the algorithm was used to predict the species of only one test
# instance. However, it can be implemented for a test dataset.
# In that case, the accuracy can be evaluated using KNN.accuracy()

print(f'For the given features: {test_row}, \nThe predictionn is:', result)


