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
# Here we will calculate the cost of a particular choice of Theta using the least squares method WITHOUT using 
# egularization (more on that stuff later ---vectorization will increase the efficiency of this algorithm). Do use 
# vectorization.


   
   
   
################################################################################################################################
 
    return cost

def gradient_descent(X, Y, Theta, alpha, num_iters):
    N = len(X) #get number of rows
    T = len(Theta)
    X = np.c_[np.ones(N), X]
    for i in range(num_iters):
    
################## Your Code Here #############################################################################################
# Here we will perform a single update to our Theta vector. Return the correct values for Theta. Note: Make sure you indent 
# properly! Use your prediction function to get the predictions.   
 
        
        #erase this: it's put here so that Python knows this is a loop
        1==1
  
 

###############################################################################################################################       
    return Theta

    
    
    
def calculate_prediction(x_val, Theta):
    y = 0 #the prediction
    
############# Your Code Here ##################################################################################################
# Calculate the predicted value of y given the feature values in x_val (which will be the features for one training example) and parameters in theta. 
# Return the correct value for y for a single example. 0 = died, 1 = lived.  
    
    
###############################################################################################################################
    return y
    
def calculate_accuracy(test_X, test_Y, Theta):
    accuracy = 0 
    
############# Your Code Here ##################################################################################################
# Calculate the percent your model correctly predicts
    
    
###############################################################################################################################
    return accuracy
