# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:38:12 2017

@author: Brilian
"""
import tensorflow as tf
import numpy as np

def L2_reg(W, beta = 1e-3):
    return beta * (W**2) / 2
    
def dropout(X,W):
    pass

def accuracy(prediction, labels):
    tot = np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))
    return 100.0 * tot / prediction.shape[0]

def MLP(X, W, b, sum_hidden, activation_func = 1):
    layer = X
    i = 0
    for i in range(sum_hidden-1):
        if activation_func != 1: layer = tf.nn.sigmoid( tf.matmul(layer, W[i]) + b[i] )
        else: layer = tf.nn.relu( tf.matmul(layer, W[i]) + b[i] )
    output = tf.matmul(layer, W[-1]) + b[-1]
    return output

def lossL2(W, beta = 1e-3):
    return tf.add_n([tf.nn.l2_loss(w) for w in W]) * beta

