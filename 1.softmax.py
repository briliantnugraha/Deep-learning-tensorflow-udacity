# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:17:42 2017

@author: Brilian
"""
import numpy as np
#plot softmax curves
import matplotlib.pyplot as plt

def softmax(x):
    expx = np.exp(x)
    return expx / expx.sum(axis=0)    
    
if __name__ == "__main__":
    scores = np.array([3.0, 1.0, 0.2])    
    print (softmax(scores))
    print (softmax(scores*10)) #the scores will be either near 1 or near 0
    print (softmax(scores/10)) #the scores will be in uniform distribution
    
    X = np.arange(-2.0, 6.0, 0.1)
    Xone = np.ones_like(X)
    scores = np.vstack([X, Xone, 0.2*Xone])
    
    plt.plot(X, softmax(scores).T, linewidth=2)
    plt.show()
    
    
    
    
    
    

