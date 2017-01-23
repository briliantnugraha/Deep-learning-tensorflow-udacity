# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:25:27 2017

@author: Brilian
"""


import _pickle as pickle
import tensorflow as tf
import numpy as np
from nn import MLP, accuracy, lossL2

def open_pickle(filepath):
    with open(filepath, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset= save['valid_dataset']
        valid_labels  = save['valid_labels']
        test_dataset = save['test_dataset'] 
        test_labels = save['test_labels']
        del save  # to free up memory
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels, image_size, num_labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def preprocess(filepath):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_pickle(filepath)    
    train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    
if __name__ == '__main__':
    image_size, num_labels, train_subset, hidden_nodes = 28, 10, 10000, 1024
    filepath = 'E:/Deep learning/notMNIST dataset/notMNIST.pickle'
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = preprocess(filepath)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    graph = tf.Graph()
    inp = input("Batch or full gradient descent [y=batch / n=full] ? ")
    activation_func = input("Pick activation function : [1. relu, 2. sigmoid] ? ")            
    batch, num_steps = 512, 100001 if inp=='y' else 1001
    
    with graph.as_default():
        #input data
        #load train, valid, and test data to tensor that attached to the graph
        if inp=='y':
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, image_size*image_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch, num_labels)) 
        else:
            tf_train_dataset, tf_train_labels = tf.constant(train_dataset[:train_subset]) , tf.constant(train_labels[:train_subset])         
        tf_valid_dataset,tf_test_dataset = tf.constant(valid_dataset), tf.constant(test_dataset)
        
        # the weight matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        W =  [ tf.Variable( tf.truncated_normal([image_size*image_size, hidden_nodes])), \
              tf.Variable( tf.truncated_normal([hidden_nodes, int(hidden_nodes/2)])), \
              tf.Variable( tf.truncated_normal([int(hidden_nodes/2), num_labels]) ) ]
        b = [ tf.Variable( tf.zeros([hidden_nodes]) ), \
            tf.Variable( tf.zeros([int(hidden_nodes/2)]) ), \
            tf.Variable( tf.zeros([num_labels]) ) ]
        #train logits
        logits = MLP(tf_train_dataset, W, b, len(W), activation_func)
        logits = tf.nn.dropout(logits, 0.5) #adding dropout
        #loss function with L2 regularization
        beta = 5e-4
        lossL2 = lossL2(W, beta)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels) + lossL2)        
        #gradient function optimizer
        backprop = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        
        #train, valid, and test prediction logits
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax( MLP(tf_valid_dataset, W, b, len(W), activation_func) )
        test_prediction = tf.nn.softmax( MLP(tf_test_dataset, W, b, len(W), activation_func) )
        
    with tf.Session(graph=graph) as session:
#        writer = tf.train.SummaryWriter( "D:/", graph = graph)
#        tf.scalar_summary('cost', loss)
#        summary_ops = tf.merge_all_summaries()
        tf.global_variables_initializer().run()
        print ("global variable initialized")
        for step in range(num_steps):
            if inp=='y':
                shape_train = train_labels.shape[0]
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step*batch) % (shape_train - batch)
                batch_data = train_dataset[offset:offset + batch]
                batch_label = train_labels[offset:offset + batch]
                if step % 100 == 0: label = batch_label 
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels: batch_label}
#                _, l, predictions   = session.run([backprop, loss, train_prediction, summary_ops], feed_dict = feed_dict)
                _, l, predictions   = session.run([backprop, loss, train_prediction], feed_dict = feed_dict)
            else:
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy array
#                _, l, predictions  = session.run([backprop, loss, train_prediction, summary_ops])
                _, l, predictions  = session.run([backprop, loss, train_prediction])
                if step % 100 == 0: label = train_labels[:train_subset]
#            writer.add_summary(summary, step)        
            if step % 100 == 0:
                print ("%d. Loss: %.2f, Training accuracy: %.2f" % (step, l, accuracy(predictions, label)) )
                print ("Validation: %.2f, Test: %.2f" % (accuracy(valid_prediction.eval(), valid_labels), \
                                                         accuracy(test_prediction.eval(), test_labels)))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph dependencies.
#        print ("Test accuracy: %.2f" % (accuracy(test_prediction.eval(), test_labels)))