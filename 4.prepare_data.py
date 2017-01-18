# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:08:23 2017

@author: Brilian
"""

#import numpy as np
import gzip, tarfile, _pickle as pickle, numpy as np
from os import listdir, stat
from os.path import isdir, join, exists
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


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
        train_folder = [path + dirs[0] + '/'+ i for i in listdir(path + dirs[0])]
        test_folder = [path + dirs[1] + '/'+ i for i in listdir(path + dirs[1])]
        return train_folder, test_folder
    except:
        print ("path is not exist or could not be opened!")

def plot_examples(train_folder):
    for i in train_folder:
        train_images = listdir(i)
        print (i)
        random_img = np.random.choice(len(train_images))
        im = Image.open(i + '/' + train_images[random_img])
        plt.imshow(im)
        plt.show()

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape,'\nMean:', np.mean(dataset),'\nStandard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    if folder[-7:] == ".pickle": continue
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
#    if exists(set_filename) and not force:
#      # You may override by setting force=True.
#      print('%s already present - Skipping pickling.' % set_filename)
#    else:
#      print('Pickling %s.' % set_filename)
#      dataset = load_letter(folder, min_num_images_per_class)
#      try:
#        with open(set_filename, 'wb') as f:
#          pickle.dump(dataset, f)#, pickle.HIGHEST_PROTOCOL)
#      except Exception as e:
#        print('Unable to save data to', set_filename, ':', e)
  return dataset_names
  
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels

if __name__ == '__main__':
    path = 'E:/Deep learning/notMNIST dataset/'
    filepath = path + 'notMNIST_large.tar.gzip'
#    extract_data(filepath)
    train_folder, test_folder = get_folder(path) #step 1
#    plot_examples(train_folder) #see example of the dataset

#     step 2
    train_datasets = maybe_pickle(train_folder, 45000)
    test_datasets = maybe_pickle(test_folder, 1800)
    
#    step 3
    train_size, valid_size, test_size = 32000, 12000, 10000    
    valid_dataset, valid_labels, train_dataset, train_labels = \
    merge_datasets(train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
    
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)
    
    #step 4
    train_datasets, train_labels = shuffle(train_dataset, train_labels)
    train_datasets, train_labels = shuffle(test_dataset, test_labels)
    train_datasets, train_labels = shuffle(valid_dataset, valid_labels)
    
    #step 5 - save to pickle file
    pickle_file = 'notMNIST.pickle'

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f)#, pickle.HIGHEST_PROTOCOL)
      f.close()
      
      statinfo = stat(pickle_file)
      print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise
      
    
    
    
            