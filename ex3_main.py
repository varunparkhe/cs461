# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:18:20 2014

@author: marksargent
"""
import pylab as pl
import numpy as np
import ex3_functions as cf
from sklearn import linear_model

filename = 'auto-mpg.data'

# Load and prepare multivariate data
'''
names info in auto-mpg.data:
    0. mpg:           continuous
    1. cylinders:     multi-valued discrete
    2. displacement:  continuous
    3. horsepower:    continuous
    4. weight:        continuous
    5. acceleration:  continuous
    6. model year:    multi-valued discrete
    7. origin:        multi-valued discrete
    8. car name:      string (unique for each instance)

'''

X = np.loadtxt(filename, usecols=(2, 3, 5))  # using displacement, horsepower, and acceleration
Y = np.loadtxt(filename, usecols=(0,))



X = cf.normalize(X)

dimX = X.shape  # get dimensions of X as a tuple (rows, columns) 
N = dimX[1]  # of columns in X; number of features
Theta = np.zeros(N + 1)  # add a column for theta0

#Our results
print "Cost before gradient descent: " , cf.calculate_cost(X, Y, Theta)
Results = cf.gradient_descent(X, Y, Theta, .01, 500)
print "Thetas: " , Results[0]
Theta = Results[0]
print "Cost after gradient descent: ", cf.calculate_cost(X, Y, Theta)


# Compare with sci-kit-learns's implementation ##########################################
# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(X, Y)

# The coefficients
print "\nResults from sci-kit-learn's linar regression method:"
print 'Coefficients: ', regr.coef_
print 'Intercept : ', regr.intercept_
# The mean square error


# Plot cost versus iterations to check if it converges
pl.xlabel("Iterations")
pl.ylabel("Cost")
pl.plot(Results[1])

pl.show()





