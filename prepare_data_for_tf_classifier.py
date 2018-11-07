#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:27:30 2018

@author: sergio
"""

import os
import numpy as np
seed = 2018
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
import cv2

NUM_INSTANCES_FRONT = 20000
NUM_INSTANCES_NO_FRONT = 20000

VALIDATION_FRONT_INSTANCES = 2000
VALIDATION_THREEQUARTER_INSTANCES = 1500
VALIDATION_PROFILE_INSTANCES = 1500





EXTENSION = '.jpg'

image_dir_path = ['/nas/datasets/vggface2_test_subset/orientation/images/front', '/nas/datasets/vggface2_test_subset/orientation/images/three-quarter_or_profile']
output_tfrecords_path = '/nas/datasets/vggface2_test_subset/orientation/tfrecords'
output_tfrecords_list = '/media/sergio/hdd/datasets/vggface2/tensorflow'

CSV_PATH = '/media/sergio/hdd/datasets/vggface2/vggface2_FULL_table.csv'

# Validation will be the already annotated
full_dataframe = pd.read_csv(CSV_PATH)

# Filter keep instances only if contains pose annotations
black_list = np.where(full_dataframe['orientation'].isnull())[0]
full_dataframe = full_dataframe.drop(black_list)
full_dataframe = full_dataframe.reset_index(drop=True)
full_dataframe = shuffle(full_dataframe, random_state = seed)
black_list = full_dataframe['class_id'] + '_' + full_dataframe['image']
black_list = black_list.get_values()
 
# Create a dataframe that contains image paths and assign a label
image_list_and_gt = []
for in_dir in image_dir_path:
    # Get all items with the configured extension
    globalList = filter(lambda element: EXTENSION in element, os.listdir(in_dir))
    globalList = np.asarray(globalList)
    for current_image_name in globalList:
        if current_image_name not in black_list:
            # Set the label to 1 if 'front', 0 otherwise
            if in_dir==image_dir_path[0]:
                label = 1
            else:
                label = 0
            
            image_list_and_gt.append({'img_path':in_dir + '/' + current_image_name, 'gt_label':label})
        else:
            print('Common element found...')


# Generate the full dataframe
dataframe = pd.DataFrame(image_list_and_gt)
dataframe = shuffle(dataframe, random_state = seed)

# Get the list of items with the different labels
front_items = np.where(dataframe['gt_label'] == 1)[0]
no_front_items = np.where(dataframe['gt_label'] == 0)[0]

# Cut it to the desired number of instances
front_items = front_items[:NUM_INSTANCES_FRONT]
no_front_items = no_front_items[:NUM_INSTANCES_NO_FRONT]

## Generate the filtered dataframe
stacked_items = np.concatenate((front_items, no_front_items), axis=0)
dataframe = dataframe.iloc[stacked_items]
dataframe = dataframe.reset_index(drop=True)

# Shuffle dataframe
dataframe = shuffle(dataframe, random_state = seed)








#######################
# Generate TF records #
#######################

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

output_tfrecords_path = '/media/sergio/hdd/datasets/vggface2/tensorflow/train_tfrecords'

# Following the dataframe annotations, generate tensorflow records
for idx in range(0, len(dataframe)):
    image_path = dataframe.at[idx, 'img_path']
    gt_label = dataframe.at[idx, 'gt_label']
    name = image_path[image_path.rfind('/')+1:-len(EXTENSION)]
    image = cv2.imread(image_path)   
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    filename = output_tfrecords_path + '/' + name + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(gt_label)),
            'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    

# Generate training, validation list
#train=dataframe.sample(frac=0.7,random_state=seed)
#validation=dataframe.drop(train.index)
train = dataframe
training_list = []
for image_path in train['img_path']:
    name = image_path[image_path.rfind('/')+1:-len(EXTENSION)]
    name = '/nas/datasets/vggface2_test_subset/orientation/tfrecords/' + name + '.tfrecords'
    training_list.append(name)
    
mylist = training_list
with open(output_tfrecords_list+'/train_list','w') as f:
    f.write( '\n'.join( mylist ) )
    
    
    

front = np.where(full_dataframe['orientation'] == 'front')[0]
threequarter = np.where(full_dataframe['orientation'] == 'three-quarter')[0]
profile = np.where(full_dataframe['orientation'] == 'profile')[0]

front = front[:VALIDATION_FRONT_INSTANCES]
threequarter = threequarter[:VALIDATION_THREEQUARTER_INSTANCES]
profile = profile[:VALIDATION_PROFILE_INSTANCES]

# Generate the filtered dataframe
stacked_items = np.concatenate((front, threequarter, profile), axis=0)
annotated_validation = full_dataframe.iloc[stacked_items]
annotated_validation = annotated_validation.reset_index(drop=True)
annotated_validation = shuffle(annotated_validation, random_state = seed)
#len(np.where(annotated_validation['orientation'] == 'profile')[0])



# Validation tfrecords
#output_tfrecords_path = '/media/sergio/hdd/datasets/vggface2/tensorflow/annotated_tfrecords'
#
#for idx in range(0, len(annotated_validation)):
#    
#    
#    
#    image_path = '/media/sergio/hdd/datasets/vggface2/ONLY_TEST/EVA_crops_vggface2/' + annotated_validation.at[idx, 'class_id'] + '/' + annotated_validation.at[idx, 'image']
#
#    orientation_raw = annotated_validation.at[idx, 'orientation']
#
#    if orientation_raw == 'front':
#        gt_label = 1
#    else:
#        gt_label = 0
#    
#    name = image_path[image_path.rfind('/')+1:-len(EXTENSION)]
#    image = cv2.imread(image_path)   
#    rows = image.shape[0]
#    cols = image.shape[1]
#    depth = image.shape[2]
#    filename = output_tfrecords_path + '/' + annotated_validation.at[idx, 'class_id'] + '_' + name + '.tfrecords'
#    print('Writing', filename)
#    writer = tf.python_io.TFRecordWriter(filename)
#    image_raw = image.tostring()
#    example = tf.train.Example(features=tf.train.Features(feature={
#            'height': _int64_feature(rows),
#            'width': _int64_feature(cols),
#            'depth': _int64_feature(depth),
#            'label': _int64_feature(int(gt_label)),
#            'image_raw': _bytes_feature(image_raw)}))
#    writer.write(example.SerializeToString())








validation_list = []
for idx in range(len(annotated_validation)): # annotated_validation['image']
    name_raw = annotated_validation['image'][idx]
    name = name_raw[:-len(EXTENSION)]
    class_id_raw = annotated_validation['class_id'][idx]
    name = '/nas/datasets/vggface2_test_subset/orientation/tfrecords/' + class_id_raw  + '_' + name + '.tfrecords'
    validation_list.append(name)

mylist = validation_list
with open(output_tfrecords_list+'/validation_list','w') as f:
    f.write( '\n'.join( mylist ) )