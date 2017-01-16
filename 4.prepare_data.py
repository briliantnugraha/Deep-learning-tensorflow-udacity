# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:08:23 2017

@author: Brilian
"""

#import numpy as np
import gzip, tarfile, _pickle as pickle

def extract_data(fileloc):
    try:
        file_tar = tarfile.open(filepath, 'r:gz')
        file_tar.extractall(path=path)
        file_tar.close()
    except:
        print ("extracting failed")
        
def get_folder(path):
    
    return 0, 0
    
if __name__ == '__main__':
    path = 'E:/Deep learning/notMNIST dataset/'
    filepath = path + 'notMNIST_large.tar.gzip'
    extract_data(filepath)
    
    train_folder, test_folder = get_folder(path)
            