#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:12:47 2018

@author: adrian
"""



file1 = open('/nas/datasets/vggface2_test_subset/orientation/validation_list','r')
tt = file1.readlines()
file2 = open('/nas/datasets/vggface2_test_subset/orientation/train_list','r')
tt2 = file2.readlines()

g = set(tt) <= set(tt2)




'/nas/datasets/vggface2_test_subset/orientation/tfrecords'
#set()