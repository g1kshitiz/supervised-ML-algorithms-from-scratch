#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kshitiz Bhandari

Implementation of k-Nearest-Neighbors from scratch for classification  
"""
from math import sqrt

# Implementing k-Nearest-Neighbors
class KNN(object):
    def __init__(self, k = 5):
        '''
        k is the number of nearest neighbors to consider (default: set to 5)
        '''
        self.k = k
    
    @staticmethod
    def euclidean_distance(row1, row2):
        '''
        Assumes row1 and row2 are rows with same number of features and
            an outcome as the last column
        Returns the n-dimensional Euclidean distance between x1 and x2
        '''
        assert len(row1) == len(row2),\
        "The number of features are different"
        # initialize
        distance_sq = 0.0
        
        # add the squared of distance of each dimension (predictor)
        # the last column is ignored (assuming the last column is the outcome)
        for i in range(len(row1) - 1):
            distance_sq += pow(row1[i] - row2[i], 2)
        # final Euclidean distance
        return sqrt(distance_sq)
    
    
    def getNeighbors(self, train_set, test_instance):
        '''
        Assumes train_set is the whole training dataset with predictors 
                and outcome as the last column
        test_instance is an instance of the test set
        Returns the most similar neighbors
        '''
        distances = list()
        # for every row in the training set
        for train_row in train_set:
            # get distance from the instance of the test_set to the current row
            # sliced up to -1 because outcome is in the last column
            dist = self.euclidean_distance(test_instance, train_row[:-1])
            # update distance to the list as a tuple
            distances.append((train_row, dist))
            
        # sort the distances (list of tuples) by the distance
        distances.sort(key = lambda x: x[1])
        
        # initialize list of k neighbors
        neighbors = []
        for i in range(self.k):
            # add the list of k nearest neighbors
            # i.e. k neighbors with the least distance
            neighbors.append(distances[i][0])
        
        return neighbors
        
        
    def predict(self, train_set, test_instance):
        # get neighbors for the current test_instance
        neighbors = self.getNeighbors(train_set, test_instance) 
        # initialize variable to keep track of classes of neighbors
        classification  = {}
        
        # determine the class of the test instance 
        # using majority from its k nearest neighbors
        for i in range(len(neighbors)):
            # outcome is in the last column of each row
            outcome = neighbors[i][-1]
            
            # update classification
            # increase count if it exists in the dictionary
            if outcome in classification:
                classification[outcome] += 1
            # otherwise store it and set its count to 1
            else:
                classification[outcome] = 1
        
        # sort the classes in descending order of their counts
        sorted_classes = sorted(classification.items(), key = lambda x: x[1], reverse = True)
        # it stores it as tuples of (class, count)
        
        # return the actual class with the highest count among its neighbors
        return sorted_classes[0][0]
    
    @staticmethod
    def accuracy(y_actual, y_hat):
        '''
        Calculates accuracy using true outcomes and predicted outcomes
        '''
        assert len(y_actual) == len(y_hat),\
        "The number of actual and predicted outcomes do not match"
        
        # initialize number of correct predicted outcomes
        num_correct = 0
        
        # iterate over both arrays simultaneously
        for actual, predicted in zip(y_actual, y_hat):
            # increase the number of predicted outcomes  by 1
            # if the prediction matches the actual outcome
            if actual == predicted:
                num_correct += 1
        
        # return the basic accuracy
        return num_correct/len(y_actual)

        