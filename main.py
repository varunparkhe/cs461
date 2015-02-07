# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:18:20 2014
@author: marksargent
"""

import numpy as np
import cost_function as cf

#Load and clean training data file for Titanic, as you did in the first Titanic exercise. Choose 
#at least 3 features to fit

#here, add code to split data into training and test sets

# Add feature scaling to keep learning rate and number of iterations reasonable here. 


Theta = np.zeros((2,1))

print "Cost before gradient descent: " , cf.calculate_cost(X_norm, Y, Theta)
print "Thetas: " , cf.gradient_descent(X_norm, Y, Theta, .01, 1000) #You may fiddle with these parameters
print "Cost after gradient descent: ", cf.calculate_cost(X_norm, Y, Theta)

##### get and print accuracy results here
#Note, you will have to modify the data going into your prediction function to revert the feature scaling 
#you did earlier
