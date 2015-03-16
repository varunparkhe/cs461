import numpy as np
import scipy.io as so
import scipy.optimize as opt


##### Function Definitions #################################################################
def costFunction(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    
    length1 = (input_layer_size+1)*(hidden_layer_size)
  
    nn1 = nn_params[:length1]
    T1 = nn1.reshape((hidden_layer_size, input_layer_size+1))
    nn2 = nn_params[length1:]
    T2 = nn2.reshape((num_labels, 1+ hidden_layer_size))      
    m = X.shape[0]# number of training examples, useful for calculations
       
    ## You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(T1.shape)
    Theta2_grad = np.zeros(T2.shape)
    
     
####################### YOUR CODE HERE!!!! ###################################################
    '''
    Instructions: implement forward and backward propogation, set J, Theta1_grad and
    Theta2_grad to the cost, first matrix of thetas and second matrix of thetas. Use
    regularization for the cost function and gradients. 
    
    For the y values, convert them to arrays, with a '1' in the zero index corresponding to a digit
    value of 1 and a '1' in 9th index corresponding to a digit value of 10, with zeros in all
    other indices.
    '''
    # From input to hidden layer
    # First get all the z values for the units in the second layer
    # Add a column of ones to X
    X = np.c_[np.ones(m), X]
    z2 = np.dot(T1 ,X.T )
    a2 = sigmoid(z2)
   
    #from hidden layer to output layer
    ones = np.ones((1,m))
    a2 = np.append(ones, a2, axis=0)
    z3 = np.dot(T2,a2)
    a3 = sigmoid(z3)
    
    cost = 0
    Delta1 = np.zeros(T1.shape)
    Delta2 = np.zeros(T2.shape)
    
    # Calculate cost and gradient ################################################################
    for i in range(m):
        #cost (except divide by 1.0/m and regularization: do this outside the loop)
        #convert int to y-vector of zeros and 1's
        y = np.zeros(num_labels)
        
        for n in range(num_labels):
            if (n == Y[i]%10):
                y[n-1] = 1
               
            
        c_i = (-y) * np.log(a3[:, i]) - (1-y) * np.log(1-a3[:,i])
        cost = cost + c_i
        
        #Gradient ################################################################################
        #deltas for last layer
        
        delta3 = a3[:, i] - y
        delta2 = np.dot(T2.T, delta3) 
        delta2 = delta2[1:] * sigmoidGradient(z2[:, i])                 
      
        Delta2 = Delta2 + np.outer(delta3, a2[:, i])
        Delta1 = Delta1 + np.outer(delta2, X[i])

        
    Theta1_grad = Delta1/m
    Theta2_grad = Delta2/m
    
    ## Add Regularization
    T1 = np.delete(T1, (0), axis=1).flatten()  
    T2 = np.delete(T2, (0), axis=1).flatten()
    sThetas = np.sum(np.square(T1)) + np.sum(np.square(T2))
    
    J = np.sum(cost) * 1.0/m  + (lambd/(2.0 * m))*sThetas
    

###############################################################################################
    # unroll gradients and concatenate    
    grad = np.concatenate([Theta1_grad.flatten(), Theta2_grad.flatten()])
    # return variables
    return J, grad  

def gradApprox(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd):
    epsilon = 0.0001
    
   
    gradientApprox = np.zeros(nn_params.size)
    for i in range(nn_params.shape[0]):
        
        nn_params1 = np.copy(nn_params) 
        nn_params2 = np.copy(nn_params)
  
        nn_params1[i] += epsilon
        nn_params2[i] -= epsilon
        cost_plus = costFunction(nn_params1, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]     
        cost_minus = costFunction(nn_params2, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]            
        cost_diff = (cost_plus - cost_minus)/(2*.0001)
     
        gradientApprox[i] = cost_diff    
        
    return gradientApprox

    
def sigmoid(h):
    sigmoid = 0
###################### YOUR CODE HERE!!!! #####################################################
    '''
    Instructions: implement the sigmoid function, return the proper value for sigmoid
    '''
    den  = 1.0 + np.exp(-1 * h)

    sigmoid = 1.0/den  
##############################################################################################
    return sigmoid
    

    
def sigmoidGradient(z):
    sigmoidGrad = 0
###################### YOUR CODE HERE!!!! #####################################################
    '''
    Inustructions: implement the first derivative (sigmoid gradient) of the sigmoid -- see pdf
    for the formula. You will need to use this in your backward propogation alogrithm. Hint, it
    is pretty simple; one line of code should do. Return the proper value of sigmoidGrad
    '''
    sigmoidGrad =  sigmoid(z) * (1 - sigmoid(z))

###############################################################################################
    return sigmoidGrad
    

def forwardPropAndAccuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y):
    print "i, h: ", input_layer_size, hidden_layer_size
    length1 = (input_layer_size+1)*(hidden_layer_size)
  
    nn1 = nn_params[:length1]
    T1 = nn1.reshape((hidden_layer_size, input_layer_size+1))
    nn2 = nn_params[length1:]
    T2 = nn2.reshape((num_labels, 1+ hidden_layer_size))      
    m = X.shape[0]# number of training examples, useful for calculations
  
    
    X = np.c_[np.ones(m), X]
    z2 = np.dot(T1 ,X.T )
    a2 = sigmoid(z2)
   
    #from hidden layer to output layer
    ones = np.ones((1,m))

    a2 = np.append(ones, a2, axis=0)
    z3 = np.dot(T2,a2)
    a3 = sigmoid(z3)
    
    predictions = np.zeros(len(Y))
    compare = np.zeros((len(predictions), 3))
    
    predictions = a3.argmax(axis=0)+1
    
    for i in range(len(predictions)):
        compare[i] = [predictions[i], Y[i], predictions[i]==Y[i]]
    

    accuracy = 0
    
    accuracy = np.sum(compare[:, 2])/len(compare)
    

    return predictions, compare, accuracy


def randomInitializeWeights(weights, factor):
##### This is implemented for you. Think: Why do we do this? #################################
 
    W = np.random.random(weights.shape)
    #normalize so that it spans a range of twice epsilon
    W = W * 2 * factor # applied element wise
    #shift so that mean is at zero  
    W = W - factor#L_in is the number of input units, L_out is the number of output 
    #units in layer
    
    return W

###############################################################################################
# helper methods
def getCost(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    cost = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]
    return cost
   

def getGrad(nn_params, *args): 
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    return costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd )[1]

###############################################################################################



###############################################################################################    
##### Start Program! ##########################################################################
###############################################################################################
    

print "Loading Saved Neural Network Parameters..."

data = so.loadmat('ex4data1.mat')

X = data['X']
Y = data['y']

#previously determined weights to check
weights = so.loadmat('ex4weights.mat')

weights1 = weights['Theta1']
weights2 = weights['Theta2']

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
lambd = 0
params = np.concatenate([weights1.flatten(), weights2.flatten()])

j, grad = costFunction(params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)

print "Cost at parameters loaded from ex4weights.mat. (This value should be about 0.383770): ", j

print "signmoidGrad of 0 (should be 0.25): ", sigmoidGradient(0)

params_check = randomInitializeWeights(np.zeros(params.shape), 15)

grad_check = costFunction(params_check[:35], 4, 4, 3, X[:10, :4], Y[:10, :], lambd)[1]

grad_approx =  gradApprox(params_check[:35], 4, 4, 3, X[:10, :4], Y[:10, :], lambd)
checkGradient = np.column_stack((grad_check, grad_approx))

print "Gradient check: the two columns should be very close: ", checkGradient

nn_params = randomInitializeWeights(np.zeros(params.shape), .12)

args = (input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)
gradient = None

count = 0
cost_count = 0
result = np.zeros(params.shape)
result = opt.fmin_cg(getCost, nn_params, fprime=getGrad, args = args, maxiter = 50)

predictions, accuracy = forwardPropAndAccuracy(result, input_layer_size, hidden_layer_size, num_labels, X, Y)[1:]

print "predictions: ", forwardPropAndAccuracy(result, input_layer_size, hidden_layer_size, num_labels, X, Y)[1]
print "accuracy: ", accuracy 
