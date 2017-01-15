# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:03:09 2017

@author: Brilian
"""
import numpy as np
def softmax(x): #get the softmax
    expx = np.exp(x)
    return expx / expx.sum(axis=0)

def one_hot():
    temp = np.zeros([3,3])
    for i in range(3): temp[i][i] = 1
    return temp
#to make the class like this:
#[[ 1.  0.  0.]
# [ 0.  1.  0.]
# [ 0.  0.  1.]]
    
def cross_entropy(X, label):
    return -label.dot(np.log(X)) # -sigma(label * log(softmax_result))
#    return - X * np.log(label)
if __name__ == "__main__":
    label = one_hot()
    scores = np.array([3.0, 1.0, 0.2])
    softmax_result = softmax(scores)
    cross_entropy_result = cross_entropy(scores, label)
    #the sum of the cross entropy is called as loss function
    #it can be used to adjust the weights and biases by minimizing the loss function 
    #(using gradient descent)
    print (cross_entropy_result)
    