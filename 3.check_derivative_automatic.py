# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:31:26 2017

@author: Brilian
"""

import numpy as np
#plot softmax curves
import matplotlib.pyplot as plt
from scipy.misc import derivative

def square(x):
    return 6*(x**5)
    
def actual_der(x):
    return 30*(x**4)
    
def check_derivative():
    x = np.arange(-10,10, 0.01).astype(np.int32)
    derivative_estimate = derivative(square, x)
    der_actual = actual_der(x)
#    derivative_x = derivative(square, X, dx=1e-6)
    
    cek= square(x)
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(x, cek, 'g.', label='Base') # red x
    plt.plot(x, der_actual, 'rx', label='Actual') # red x
    plt.plot(x, derivative_estimate, 'b-', label='Estimate') # blue +
    plt.legend(loc=9)
    plt.show()
        
if __name__ == "__main__":
    check_derivative()