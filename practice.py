# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 20:27:18 2015

@author: Mark
"""
#the import with an alias
import numpy as np

# creatint a numpy array
a = np.array([1, 4, 5, 8], float)

#slicing operations for lists also work for numpy arrays 
a[:2]

#creating a 2d numpy matrix
a = np.array([[1, 2, 3], [4, 5, 6]], float)

a[0,0]
a[0,1]

#slicing matrices
#get row 1
a[1,:]

#get column 2
a[:,2]

#backward indices used in slicing
a[-1:,-2:]

#the shape property
a.shape

#the in operation
2 in a

#reshaping arrays
a = np.array(range(10), float) #make an array with values 0-9

#change it to a 5x2 matrix
a = a.reshape((5, 2))

#create a list from an numpy array
a.tolist()

#fill an array with a single value
a = np.array([1, 2, 3], float)
a.fill(0)

#transposition
a = np.array(range(6), float).reshape((2, 3))#create a vector, then reshape

a.transpose()

#convert matrix to vector
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a.flatten()

#concatenate vectors
a = np.array([1,2], float)
b = np.array([3,4,5,6], float)
c = np.array([7,8,9], float)
np.concatenate((a, b, c))

#concatenate matrices
a = np.array([[1, 2], [3, 4]], float)
b = np.array([[5, 6], [7,8]], float)

np.concatenate((a,b), axis=0)
np.concatenate((a,b), axis=1)

#element-wise mathematical operations
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
a + b
a - b
a * b
b / a
a % b
b**a

#iterating over arrays

#It is possible to iterate over arrays in a manner similar to that of lists:
a = np.array([1, 4, 5], int)
for x in a:
    print x


#For multidimensional arrays, iteration proceeds over the first axis such that each loop returns a
#subsection of the array:
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for x in a:
    print x

#Multiple assignment can also be used with array iteration:
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for (x, y) in a:
    print x * y

#dot product of vectors
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
np.dot(a, b)

# dot product of matrices
a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)

np.dot(b, a)
np.dot(a, c)
np.dot(c, a)
