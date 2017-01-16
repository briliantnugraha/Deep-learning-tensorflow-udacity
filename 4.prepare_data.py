# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:08:23 2017

@author: Brilian
"""

#import numpy as np
import gzip, tarfile, _pickle as pickle, numpy as np
from os import listdir
from os.path import isdir
from PIL import Image
import matplotlib.pyplot as plt

def extract_data(fileloc):
    try:
        file_tar = tarfile.open(filepath, 'r:gz')
        file_tar.extractall(path=path)
        file_tar.close()
    except:
        print ("extracting failed")
        
def get_folder(path):
    try:
        dirs = listdir(path)
        dirs = [i for i in dirs if isdir(path + i)]
        train_folder = [dirs[0] + '/'+ i for i in listdir(path + dirs[0])]
        test_folder = [dirs[1] + '/'+ i for i in listdir(path + dirs[1])]
        return train_folder, test_folder
    except:
        print ("path is not exist or could not be opened!")

def plot_examples(train_folder):
    for i in train_folder:
        train_images = listdir(path + i)
        print (i)
        random_img = np.random.choice(len(train_images))
        im = Image.open(path + i + '/' + train_images[random_img])
        plt.imshow(im)
        plt.show()
        
if __name__ == '__main__':
    path = 'E:/Deep learning/notMNIST dataset/'
    filepath = path + 'notMNIST_large.tar.gzip'
#    extract_data(filepath)
    
    train_folder, test_folder = get_folder(path)
    
    plot_examples(train_folder) #see example of the dataset
    
    
            