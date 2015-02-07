import csv as csv 
import numpy as np
 
### Functions #########################################################################   
def calculate_cost(X, Y, Theta):
    cost = 0.0 
################## Your Code Here #############################################################################################
# Calculate cost here. Use vectorization.

   
   
   
################################################################################################################################
    
    return cost

  
        
        
def sigmoid(X):
    d=0
################## Your Code Here #############################################################################################
# Plug 'X' into the sigmoid function and return result. 


###############################################################################################################################       
    return d
    
def gradient_descent(X, Y, Theta, alpha, num_iters):
    N = len(X) #get number of rows
    T = len(Theta)
    X = np.c_[np.ones(N), X]
    for i in range(num_iters):
    
################## Your Code Here #############################################################################################
# Here we will perform a single update to our Theta vector. Return the correct values for Theta. Note: Make sure you indent 
# properly! Use your prediction function to get the predictions. Use vectorization.
# X represents values for features, Y represents actual outcomes, Theta represents the parameters. Alpha represents the learning rate
# num_iters represents how many times updates will be performed. 
 
        
        #erase this: it's put here so that Python knows this is a loop
        1==1
  
 

###############################################################################################################################       
    return Theta
    
    
def get_accuracy(test_X, test_Y, Theta):
  percent_right = 0
################## Your Code Here #############################################################################################
# Calculate the % your model gets right. 

   
   
   
################################################################################################################################
  return percent_right  

def predict(X_example, actual_Y, Theta):
    survived = 0
################## Your Code Here #############################################################################################
# Make a prediction for a single example (did they survive 1 for yes, 0 for no?). 

   
  
   
################################################################################################################################    
    return survived
#### Program ############################################################################
#########################################################################################
# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next()  

# Preprocess the data
data = []
for row in csv_file_object:
    data.append(row)
    
data = np.array(data)



'''
Example code for preprocessing a selection of data columns, creating Theta, etc: feel free to use or modify
data = data[:, [1, 2, 4, 5]]
data = data[np.all(data != '', axis=1)]
data[data == 'male'] = '1'
data[data == 'female'] = '0'

data = data.astype(np.float)



Y = data[:, 0]
X = data[:, 1:4]
Theta = [0.5,0.5,0.5,0.5]

'''
################## Your Code Here #############################################################################################
'''
Calculate the cost on the training set before gradient descent and print. Run gradient descent. Print Theta values. 
Calculate the cost on the training set after running gradient descent. 




###############################################################################################################################



##
