# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:50:13 2014

@author: marksargent
"""
import numpy as np

def calculate_cost(X, Y, Theta):
    cost = 0.0 #You must return the correct value for cost
    
    #add y-intercept term with a column of ones
    dimX = X.shape #get dimensions of X as a tuple (rows, columns) 
    N = dimX[0] #get number of rows
    X = np.c_[np.ones(N), X]#add column of ones at beginning to accommodate theta0
    

################## Your Code Here #############################################################################################
# Here we will calculate the cost of a particular choice of Theta using the least squares method . Use vectorization. Make sure
# it can handle any number of features

  
  
################################################################################################################################
 
    return cost

def gradient_descent(X, Y, Theta, alpha, num_iters):
    N = len(X) #get number of rows
    T = len(Theta)
    X_ones = np.c_[np.ones(N), X]#add column of 1s
    Costs = np.zeros(num_iters)

    
    for i in range(num_iters):   
################## Your Code Here #############################################################################################
# Here we will perform a single update to our predictions vector. Note: Make sure you indent 
# properly! This function returns both the Theta vector, and a Costs vector that keeps track of the cost for each iteration. 
        print "delete this line please"
       
       
###############################################################################################################################  
        
        
         
        
    return Theta, Costs

    
def normalize(X):
################## Your Code Here #############################################################################################
# Perform mean normalization and feature scaling, using standard deviation. Return right value of X

    
       
       
###############################################################################################################################       
    return X
