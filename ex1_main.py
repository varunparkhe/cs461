# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:18:20 2014

@author: marksargent
"""
import pylab as pl
import numpy as np
import cost_function as cf

filename = 'auto-mpg.data'

#Load and prepare data
X = np.loadtxt(filename, usecols=(4,))
Y = np.loadtxt(filename, usecols=(0,))

# Added feature scaling to keep learning rate and number of iterations reasonable. 
X_norm = (X - np.mean(X))/np.std(X)

Theta = np.zeros((2,1))

print "Cost before gradient descent: " , cf.calculate_cost(X_norm, Y, Theta)
print "Thetas: " , cf.gradient_descent(X_norm, Y, Theta, .01, 1000)
print "Cost after gradient descent: ", cf.calculate_cost(X_norm, Y, Theta)

#plot data
pl.xlabel("Weight of Car")
pl.ylabel("MPG")

b = Theta[0]
m = Theta[1]

# only plot regression line if m and b are not zero, that is, if the functions in cost_function are implemented
if (b != 0 and m != 0):
    
    #b and m are calculated for a scaled X, need to reconvert to graph and predict
    b -= np.mean(X)/np.std(X) * m
    m = m/np.std(X)

    # make a line to plot
    yp = pl.polyval([m,b],X)

    # plot line and scatter diagram
    pl.plot(X, yp)

# plot scatter diagram anyway
pl.scatter(X, Y)
pl.show()
