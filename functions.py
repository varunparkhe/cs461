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
# vectorization or regularization (more on that stuff later ---vectorization will increase the efficiency of this algorithm). 
# You will basically use a for loop and calculate the predicted value (Use the calculate_prediction function you completed)
# value for a row in the X matrix, subtract the corresponding actual value in Y, square the result, and add to a running
# sum. 

    predictions = X.dot(Theta) 
    
    sq_errors = (predictions - Y)**2
           
    #Divide by twice the number of examples
    cost = sq_errors.sum()* 1.0/(2.0*N)
################################################################################################################################
 
    return cost

def gradient_descent(X, Y, Theta, alpha, num_iters):
   #Keep doing update_Theta until difference in cost function is below a threshold.  
    N = len(X) #get number of rows
    T = len(Theta)
    X = np.c_[np.ones(N), X]
    for i in range(num_iters):
    
################## Your Code Here #############################################################################################
# Here we will perform a single update to our Theta vector. Return the correct values for Theta. Note: Make sure you indent 
# properly! Use your prediction function to get the predictions.   
 
        predictions = X.dot(Theta).flatten()
        
        for k in range(T):
 
            errors = (predictions - Y) * X[:, k]

            Theta[k][0] = Theta[k][0] - alpha * (1.0 / N) * errors.sum()
  
 

###############################################################################################################################       
    return Theta

    
    
    
def calculate_prediction(x_val, Theta):
    y = 0 #the prediction
    
############# Your Code Here ##################################################################################################
# Calculate the predicted value of y given the feature values in x_val (which will be the features for one training example) and parameters in theta. 
# Return the correct value for y.   
    
    
###############################################################################################################################
    return y