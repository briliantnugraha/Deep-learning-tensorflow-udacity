# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:57:25 2017

@author: Brilian
"""

import _pickle as pickle, tensorflow as tf, numpy as np
from nn import accuracy
import matplotlib.pyplot as plt

def open_pickle(path):
    with open(path, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels, channels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def prepare_tf_data(batch, image_size, channels, num_labels, valid_dataset, test_dataset):
    #prepare tensor for dataset and labels
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, image_size, image_size, channels))
    tf_train_label = tf.placeholder(tf.float32, shape = (batch, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset.reshape([-1,image_size, image_size, channels]))
    tf_test_dataset = tf.constant(test_dataset.reshape([-1,image_size, image_size, channels]))
    return tf_train_dataset, tf_train_label, tf_valid_dataset, tf_test_dataset

def conv2d(X, W, b, stride=1):
    return tf.nn.relu( tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME') + b )

#this our CNN model (2 conv layer, 1 reshape, 1 fully connected layer)
def model(X, W, b, convnet_layer):
    layer = X
    
    #this is 2 layer conv net
    for i in range(convnet_layer):
        layer = conv2d(layer, W[i], b[i], stride= 1)
        layer = maxpool2d(layer, k=2)
    shape = layer.get_shape().as_list() #reshape
    pool_layer = tf.reshape(layer, [shape[0], shape[1] * shape[2] * shape[3]])
#    print (shape, W[0].get_shape(), W[i].get_shape(), W[i+1].get_shape(), W[i+2].get_shape(), pool_layer.get_shape())
    layer = tf.nn.relu( tf.matmul(pool_layer, W[i+1]) + b[i+1] )
    layer = tf.nn.dropout(layer, 0.9)
    return tf.matmul( layer, W[i+2] ) + b[i+2]

def maxpool2d(X, k = 2):
    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

if __name__ == "__main__":
    batch, image_size, channels = 100, 28, 1
    patch_size, depth, num_hidden, num_steps, num_labels = 5, 32, 1024, 2001, 10
    
    path = 'E:/Deep learning/notMNIST dataset/notMNIST.pickle'
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_pickle(path)    
    train_dataset, train_labels = reformat(train_dataset, train_labels, channels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, channels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, channels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    total_train = train_labels.shape[0]        

    #for graph preparation
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset, tf_train_label, tf_valid_dataset, tf_test_dataset = \
            prepare_tf_data(batch, image_size, channels, num_labels, valid_dataset, test_dataset)
        
        #prepare the weights and biases
        W = [ tf.Variable(tf.truncated_normal([patch_size, patch_size, channels, depth], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth*2], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([int(image_size // 4 * image_size // 4 * depth * 2), num_hidden], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))   ]
        
        b = [ tf.Variable(tf.zeros([depth])), \
              tf.Variable(tf.constant(1.0, shape=[depth*2])), \
              tf.Variable(tf.constant(1.0, shape=[num_hidden])), \
              tf.Variable(tf.constant(1.0, shape=[num_labels]))  ]
        #training logits and loss function
        logits = model(tf_train_dataset, W, b, 2)
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_label) )
        
        #optimizer (gradient descent)
        learning_rate = 0.01
        global_step = tf.Variable(0)
        learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=1000, decay_rate=0.6)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(loss)
        
        #train, valid, and testing prediction
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, W, b, 2))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, W, b, 2))
        
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("global variable initialized")
        
        losses = []
        for step in range(num_steps):
            offset = (batch*step) % (total_train - batch)
            batch_data = train_dataset[offset:offset+batch, :, :, :]
            batch_label = train_labels[offset:offset+batch, :]
            feed_dict = { tf_train_dataset: batch_data, tf_train_label: batch_label }
            _, l, prediction = session.run([optimizer,loss, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                losses.append(l)
                print ("%d. loss: %.2f, train accuracy: %.2f" % (step, l, accuracy(prediction, batch_label)))
                print ("valid accuracy: %.2f" % ( accuracy(valid_prediction.eval(), valid_labels) ) )
                print ("test accuracy: %.2f" % ( accuracy(test_prediction.eval(), test_labels) ) )
        plt.plot(losses)
        plt.show()