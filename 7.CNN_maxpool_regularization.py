# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:57:25 2017

@author: Brilian
"""

import _pickle as pickle, tensorflow as tf, numpy as np
from nn import accuracy, lossL2

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

#this will format the dataset from 3d to 4d, for train: (200000, 28,28) into (200000, 28,28, 1) since the image is grayscale
#for the label, it will be turned into one hot encoding class
def reformat(dataset, labels, channels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

#prepare tensor for train, valid, and test dataset and labels
def prepare_tf_data(batch, image_size, channels, num_labels, valid_dataset, test_dataset):
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, image_size, image_size, channels))
    tf_train_label = tf.placeholder(tf.float32, shape = (batch, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset.reshape([-1,image_size, image_size, channels]))
    tf_test_dataset = tf.constant(test_dataset.reshape([-1,image_size, image_size, channels]))
    return tf_train_dataset, tf_train_label, tf_valid_dataset, tf_test_dataset

#this our CNN model (2 conv layer, 1 reshape, 1 fully connected layer)
#the first and second net will be used as conv. layer (batch, 28 , 28, 1) - total data, image height, image width, rgb / grayscale
#after conv. matrix has been done in each conv. layer, then we will do max pooling 2x2 (this will reduce size of image)
#into : (batch, 14, 14, 16), conv. layer 2,(batch, 7, 7, 32), conv. layer 2
#on layer 3, the layer will be reshape into (batch, 7*7*32), and fully connected layer will be done in this layer
#(batch, 7*7*32) conv with (7*7*7*32, 64)
#the output layer (batch, 64) * (64, num_labels), 
def model(X, W, b, convnet_layer):
    layer = X
    
    #this is 2 layer conv net
    for i in range(convnet_layer):
        logits = tf.nn.conv2d(layer, W[i], [1, 2, 2, 1], padding='SAME') + b[i]
        layer = tf.nn.relu( logits )
#        layer = tf.nn.dropout(layer, 0.5)
    #reshape
    shape = layer.get_shape().as_list()
    layer_reshape = tf.reshape(layer, [shape[0], shape[1] * shape[2] * shape[3]])
    layer = tf.nn.relu( tf.matmul(layer_reshape,W[i+1]) + b[i+1])
    layer = tf.nn.dropout(layer, 0.85)
    return tf.matmul( layer, W[i+2] ) + b[i+2]

if __name__ == "__main__":
    #initialize total batch, image size, and channel (grayscale = 1)
    batch, image_size, channels = 128, 28, 1
    #initialize patch size for conv network, depth of conv net, total hidden nodes in fully connected layer
    #number of steps for looping for each epoch, and total labels (i.e. 10 labels)
    patch_size, depth, num_hidden, num_steps, num_labels = 5, 32, 64, 2001, 10
    
    #get the pickle using the path given, and take training, valid, and testing dataset and labels
    path = 'E:/Deep learning/notMNIST dataset/notMNIST.pickle'
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_pickle(path)    
    train_dataset, train_labels = reformat(train_dataset, train_labels, channels) #reformat the dataset and labels
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, channels) #reformat the dataset and labels
    test_dataset, test_labels = reformat(test_dataset, test_labels, channels) #reformat the dataset and labels
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    total_train = train_labels.shape[0]        

#    for graph preparation
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset, tf_train_label, tf_valid_dataset, tf_test_dataset = \
            prepare_tf_data(batch, image_size, channels, num_labels, valid_dataset, test_dataset)
        
#        prepare the weights and biases, there will be 3 hidden layer (2conv layer, and 1 fully connected layer)
#        and one output layer
        W = [ tf.Variable(tf.truncated_normal([patch_size, patch_size, channels, depth], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth*2], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([int(image_size // 4 * image_size // 4 * depth * 2), num_hidden], stddev=0.1)), \
              tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))   ]
        
        b = [ tf.Variable(tf.zeros([depth])), \
              tf.Variable(tf.constant(1.0, shape=[depth*2])), \
              tf.Variable(tf.constant(1.0, shape=[num_hidden])), \
              tf.Variable(tf.constant(1.0, shape=[num_labels]))  ]
        
        #training logits and loss function
        logits = model(tf_train_dataset, W, b, 2) #get teh logits result for training
        #loss function, this will be used in gradient descent to adjust weights and biases
        beta = 1e-3
        L2_reg = lossL2(W, beta)
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_label) + L2_reg)
        
        #optimizer (gradient descent)
        learning_rate = 0.01
        global_step = tf.Variable(0)
        #learning rate decay is useful for reduce the learning rate value when the vanishing gradient occurs
        learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=1000, decay_rate=0.8)
        #we will use gradient descent optimizer function to minimize the loss and adjust the weights and biases
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(loss)
        
        #train, valid, and testing prediction
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, W, b, 2))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, W, b, 2))
        
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run() #initialize all variable 
        print("global variable initialized")
        
        for step in range(num_steps):
            #make the batches, and restart the batches whenever the step* batch > total_train data
            offset = (batch*step) % (total_train - batch) 
            batch_data = train_dataset[offset:offset+batch, :, :, :] #get batch training data
            batch_label = train_labels[offset:offset+batch, :] #get batch training label
            feed_dict = { tf_train_dataset: batch_data, tf_train_label: batch_label } #put the batch into dict
            #push the dictionary to the session as the input, 
            _, l, prediction = session.run([optimizer,loss, train_prediction], feed_dict=feed_dict) 
            if step % 50 == 0:
                #for every 50 epochs, output the accuracy of training, valid, and testing
                #(to recheck whether the training model is stable or not)
                print ("%d. loss: %.2f, train accuracy: %.2f" % (step, l, accuracy(prediction, batch_label)))
                print ("valid accuracy: %.2f" % ( accuracy(valid_prediction.eval(), valid_labels) ) )
        print ("test accuracy: %.2f" % ( accuracy(test_prediction.eval(), test_labels) ) )      