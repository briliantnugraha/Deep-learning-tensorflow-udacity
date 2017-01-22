# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:09:48 2017

@author: Brilian
"""

import _pickle as pickle
import tensorflow as tf
import numpy as np

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

def accuracy(prediction, labels):
    tot = np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))
    return 100.0 * tot / prediction.shape[0]
        
if __name__ == '__main__':
    filepath = 'C:/Users/Brilian/Documents/GitHub/Deep learning tensorflow - udacity/notMNIST.pickle'
    train_dataset, train_labels, valid_dataset, \
        valid_labels, test_dataset, test_labels = open_pickle(filepath)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    image_size = 28
    num_labels = 10        
    train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels)
    
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
        
    train_subset = 10000
    graph = tf.Graph()
    inp = input("Batch or full gradient descent [y=batch / n=full] ? ")        
    batch, num_steps = 128, 3001 if inp=='y' else 801
    
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
        W, b = tf.Variable( tf.truncated_normal([image_size*image_size, num_labels]) ), tf.Variable( tf.zeros([num_labels]) )
        
        #do logistic regressiion, and use softmax for classification
        #get the loss to be used in the backprop step
        valid_prediction,test_prediction = tf.nn.softmax( tf.matmul(tf_valid_dataset, W) + b ), tf.nn.softmax( tf.matmul(tf_test_dataset, W) + b )
        
        train_logits = tf.matmul(tf_train_dataset, W) + b
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=tf_train_labels))        
        train_prediction = tf.nn.softmax(train_logits)    
        #gradient function
        backprop = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the biases
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
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels: batch_label}
                _, l, predictions = session.run([backprop, loss, train_prediction], feed_dict = feed_dict)
                if step % 100 == 0:
                    print ("Loss at step %d: %f" % (step, l))
                    print ("Training accuracy: %.2f" % (accuracy(predictions, batch_label)))
                    print ("Validation accuracy: %.2f" % (accuracy(valid_prediction.eval(), valid_labels)))
            else:
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy array
                _, l, predictions = session.run([backprop, loss, train_prediction])
                    
                if step % 100 == 0:
                    print ("Loss at step %d: %f" % (step, l))
                    print ("Training accuracy: %.2f" % (accuracy(predictions, train_labels[:train_subset])))
                    print ("Validation accuracy: %.2f" % (accuracy(valid_prediction.eval(), valid_labels)))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph dependencies.
                
        print ("Test accuracy: %.2f" % (accuracy(test_prediction.eval(), test_labels)))
                
        