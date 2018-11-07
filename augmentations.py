# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from scipy import misc
import random as rnd

import sys
import params

H_SZ = params.OUT_HEIGHT
W_SZ = params.OUT_WIDTH
R_SZ = float(W_SZ)/float(H_SZ)




def _to_grayscale(images):
    
    K_r = 0.2126729
    K_g = 0.7151522
    K_b = 0.0721750
    
    i0 = tf.slice(images, [0, 0, 0], [-1, -1, 1]) * K_r + tf.slice(images, [0, 0, 1], [-1, -1, 1]) * K_g + tf.slice(images, [0, 0, 2], [-1, -1, 1]) * K_b

    images = tf.concat(axis=2, values=[i0, i0, i0])

    return images
    
    
def to_grayscale(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _to_grayscale(images), lambda: images)
    gray_flag = tf.cond(pred, lambda: tf.identity(True), lambda: tf.identity(False))

    return images, gray_flag

    
def gamma_correct(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: images**0.5, lambda: images)
    
    return images

def _blur_image(images):
    
    kernel = tf.constant([0.0030, 0.0133, 0.0219, 0.0133, 0.0030,
                          0.0133, 0.0596, 0.0983, 0.0596, 0.0133,
                          0.0219, 0.0983, 0.1621, 0.0983, 0.0219,
                          0.0133, 0.0596, 0.0983, 0.0596, 0.0133,
                          0.0030, 0.0133, 0.0219, 0.0133, 0.0030], shape=[5, 5, 1, 1])
    
    images_ = tf.expand_dims(images, 0)
    image_channels = list()
    n_channels = 3

    for c in range(n_channels):
        i_c = tf.slice(images_, [0, 0, 0, c], [-1, -1, -1, 1])
        image_channels.append(tf.nn.conv2d(i_c, kernel, [1, 1, 1, 1], padding='SAME'))
            
    images_ = tf.concat(axis=3, values=image_channels)
    images = tf.squeeze(images_, axis=[0])
        
    return images


def blur_image(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _blur_image(images), lambda: images)

    return images    


def _flip_horizontally(images):

    return tf.image.flip_left_right(images)
    


def flip_horizontally(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _flip_horizontally(images), lambda: images)
        
    return images
    

def _lower_resolution_factor(images, factor):
    
    images = tf.image.resize_images(images, tf.constant([np.round(params.OUT_HEIGHT * factor).astype(int),
                                                         np.round(params.OUT_WIDTH * factor).astype(int)]))
    images = tf.image.resize_images(images, tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH]))
    
    return images
    
    
def _lower_resolution(images):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.25)
    
    images = tf.cond(pred,
                     lambda: _lower_resolution_factor(images, 0.5),
                     lambda: _lower_resolution_factor(images, 0.75))

    return images


def lower_resolution(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _lower_resolution(images), lambda: images)
        
    return images


def crop_ul(image, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0, 0.0, 1.0 - crop_factor, 1.0 - crop_factor], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def crop_ur(image, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0, crop_factor, 1.0 - crop_factor, 1.0], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def crop_dr(image, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([crop_factor, crop_factor, 1.0, 1.0], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def crop_dl(image, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([crop_factor, 0.0, 1.0, 1.0 - crop_factor], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image   


def _rnd_crop(images):
    
    p_order0 = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order0, 0.5)
    crop_factor = tf.cond(pred, lambda: tf.constant(0.25, dtype=tf.float32), lambda: tf.constant(0.125, dtype=tf.float32))
    
    p_order1 = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    
    pred_ul = tf.less(p_order1, 0.25)
    pred_ur = tf.logical_and(tf.greater_equal(p_order1, 0.25), tf.less(p_order1, 0.5))
    pred_dr = tf.logical_and(tf.greater_equal(p_order1, 0.5), tf.less(p_order1, 0.75))
    pred_dl = tf.greater_equal(p_order1, 0.75)
            
    images = tf.cond(pred_ul, lambda: crop_ul(images, crop_factor), lambda: images)
    images = tf.cond(pred_ur, lambda: crop_ur(images, crop_factor), lambda: images)
    images = tf.cond(pred_dr, lambda: crop_dr(images, crop_factor), lambda: images)
    images = tf.cond(pred_dl, lambda: crop_dl(images, crop_factor), lambda: images)
    
    return images    
    
    
#def rnd_crop(images, prob):
#    
#    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
#    pred = tf.less(p_order, prob)
#    images = tf.cond(pred, lambda: _rnd_crop(images), lambda: images)
#    
#    return images



def pad_ul(image, image_mean, crop_factor):

    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0, 0.0, 1.0 + crop_factor, 1.0 + crop_factor], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    extrapolation_value=image_mean
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def pad_ur(image, image_mean, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0, 0.0 - crop_factor, 1.0 + crop_factor, 1.0], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    extrapolation_value=image_mean
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def pad_dr(image, image_mean, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0 - crop_factor, 0.0 - crop_factor, 1.0, 1.0], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    extrapolation_value=image_mean
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image

def pad_dl(image, image_mean, crop_factor):
    
    images = tf.expand_dims(image, axis=0)
    box = tf.stack([0.0 - crop_factor, 0.0, 1.0, 1.0 + crop_factor], axis=0)
    boxes = tf.expand_dims(box, axis=0)
    image_crop = tf.image.crop_and_resize(
    images,
    boxes=boxes,
    box_ind=tf.constant([0], dtype=tf.int32),
    crop_size=tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH], dtype=tf.int32),
    extrapolation_value=image_mean
    )
    image = tf.squeeze(image_crop, axis=0)
    
    return image   

def _rnd_pad(images):
    
    p_order0 = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order0, 0.5)
    crop_factor = tf.cond(pred, lambda: tf.constant(0.25, dtype=tf.float32), lambda: tf.constant(0.125, dtype=tf.float32))
    
    p_order1 = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    
    pred_ul = tf.less(p_order1, 0.25)
    pred_ur = tf.logical_and(tf.greater_equal(p_order1, 0.25), tf.less(p_order1, 0.5))
    pred_dr = tf.logical_and(tf.greater_equal(p_order1, 0.5), tf.less(p_order1, 0.75))
    pred_dl = tf.greater_equal(p_order1, 0.75)
    
    images_mean = 0.5 #tf.reduce_mean(images, axis=[0, 1, 2])
    images = tf.cond(pred_ul, lambda: pad_ul(images, images_mean, crop_factor), lambda: images)
    images = tf.cond(pred_ur, lambda: pad_ur(images, images_mean, crop_factor), lambda: images)
    images = tf.cond(pred_dr, lambda: pad_dr(images, images_mean, crop_factor), lambda: images)
    images = tf.cond(pred_dl, lambda: pad_dl(images, images_mean, crop_factor), lambda: images)
    
    return images  


def _sim_shitty_box(image):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.66)
    image = tf.cond(pred, lambda: _rnd_crop(image), lambda: _rnd_pad(image))
    
    return image
    
def sim_shitty_box(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _sim_shitty_box(images), lambda: images)
    
    return images
    
def _rnd_patch(image):
    
    h_patch = params.OUT_HEIGHT / 3
    w_patch = params.OUT_WIDTH / 3
    mask = tf.ones([h_patch, w_patch, 3], dtype=tf.float32)

    h_offset = tf.random_uniform(shape=[], minval=0, maxval=params.OUT_HEIGHT-h_patch-1, dtype=tf.int32)
    w_offset = tf.random_uniform(shape=[], minval=0, maxval=params.OUT_WIDTH-w_patch-1, dtype=tf.int32)
    mask1 = tf.image.pad_to_bounding_box(mask, offset_height=h_offset, offset_width=w_offset, target_height=params.OUT_HEIGHT, target_width=params.OUT_WIDTH)
    mask0 = mask1 - 0.5
    mask0 = -mask0
    mask0 += 0.5

    image = image * mask0 + 0.5 * mask1
    
    return image

def rnd_patch(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _rnd_patch(images), lambda: images)
    
    return images

def _change_illuminant_blue(images):
    
    i0_r = tf.slice(images, [0, 0, 0], [-1, -1, 1])
    i0_g = tf.slice(images, [0, 0, 1], [-1, -1, 1])
    i0_b = tf.slice(images, [0, 0, 2], [-1, -1, 1]) * 0.8 + 0.2

    images = tf.concat(axis=2, values=[i0_r, i0_g, i0_b])
    
    return images
    

def _change_illuminant_orange(images):
    
    i0_r = tf.slice(images, [0, 0, 0], [-1, -1, 1])
    i0_g = (1.0/3.0) * tf.slice(images, [0, 0, 1], [-1, -1, 1]) * 0.5412 + (2.0/3.0) * tf.slice(images, [0, 0, 1], [-1, -1, 1])
    i0_b = (1.0/3.0) * tf.slice(images, [0, 0, 2], [-1, -1, 1]) * 0.1255 + (2.0/3.0) * tf.slice(images, [0, 0, 2], [-1, -1, 1])

    images = tf.concat(axis=2, values=[i0_r, i0_g, i0_b])

    return images


def _change_illuminant(images):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)

    images = tf.cond(pred, lambda: _change_illuminant_blue(images), lambda: _change_illuminant_orange(images))
    
    return images


def change_illuminant(images, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    images = tf.cond(pred, lambda: _change_illuminant(images), lambda: images)
    
    return images
    

def _add_chroma_noise(image):
    
    STD_RED = 0.11
    STD_GREEN = 0.08
    STD_BLUE = 0.15

    n_red = tf.random_normal([1, params.OUT_HEIGHT / 2, params.OUT_WIDTH / 2, 1], mean=1.0, stddev=STD_RED)
    n_red = tf.image.resize_nearest_neighbor(n_red, [params.OUT_HEIGHT, params.OUT_WIDTH])
    n_green = tf.random_normal([1, params.OUT_HEIGHT / 2, params.OUT_WIDTH / 2, 1], mean=1.0, stddev=STD_GREEN)
    n_green = tf.image.resize_nearest_neighbor(n_green, [params.OUT_HEIGHT, params.OUT_WIDTH])
    n_blue_low = tf.random_normal([1, params.OUT_HEIGHT / 8, params.OUT_WIDTH / 8, 1], mean=1.0, stddev=STD_BLUE)
    n_blue_low = tf.image.resize_nearest_neighbor(n_blue_low, [params.OUT_HEIGHT, params.OUT_WIDTH])
    n_blue_high = tf.random_normal([1, params.OUT_HEIGHT / 2, params.OUT_WIDTH / 2, 1], mean=1.0, stddev=STD_BLUE)
    n_blue_high = tf.image.resize_nearest_neighbor(n_blue_high, [params.OUT_HEIGHT, params.OUT_WIDTH])
    n_blue = 0.5 * n_blue_low + 0.5 * n_blue_high
    
    n_all = tf.concat([n_red, n_green, n_blue], axis=3)
    n_all = tf.squeeze(n_all, axis=0)

    image *= n_all
    image = tf.minimum(image, 1.0)
    
    return image


def add_chroma_noise(image, prob):
    
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, prob)
    image = tf.cond(pred, lambda: _add_chroma_noise(image), lambda: image)
    
    return image

    
def data_augment(images):
    
    # to grayscale
    # images, gray_flag = to_grayscale(images, params.DATA_AUG_P_gray)
    
    # gamma correction
    images = gamma_correct(images, params.DATA_AUG_P_gamma)
    
    # blurring
    images = blur_image(images, params.DATA_AUG_P_blur)
    
    # horzontal flipping
    images = flip_horizontally(images, params.DATA_AUG_P_flip)
    
    # changing illuminant (only if grayscale has not triggered)
    images = change_illuminant(images, params.DATA_AUG_P_light)
    
    # adding chroma noise
    images = add_chroma_noise(images, params.DATA_AUG_P_chroma)
    
    # lowering resolution
    images = lower_resolution(images, params.DATA_AUG_P_lowres)
    
    # simulating shitty bounding boxes
    images = sim_shitty_box(images, params.DATA_AUG_P_shittybox)
    
    # putting a random 0.5 patch on the image
    images = rnd_patch(images, params.DATA_AUG_P_rndpatch)

    return images    




def do_whitening(image):
    return tf.image.per_image_standardization(image)


def _read_and_decode_png(filename_queue, input_size=[480, 640, 3]):

    reader = tf.WholeFileReader()
    name, image_file = reader.read(filename_queue)
    image = tf.image.decode_png(image_file)
    image.set_shape(input_size)

    return image
    


def _read_and_decode_trainval(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
            'name_raw': tf.FixedLenFeature([], tf.string),
            })

    image_ = tf.decode_raw(features['image_raw'], tf.uint8)
    image_.set_shape(params.OUT_HEIGHT_ORIG*params.OUT_WIDTH_ORIG*3)
    image_ = tf.reshape(image_, (params.OUT_HEIGHT_ORIG,params.OUT_WIDTH_ORIG,3))
    image = tf.cast(image_, tf.float32)/255.0
    
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape(params.NUM_CLASSES)
    
    name = features['name_raw']
    
    return image, image_, label, name
    

def _read_and_decode_test(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'name_raw': tf.FixedLenFeature([], tf.string),
            })

    image_ = tf.decode_raw(features['image_raw'], tf.uint8)
    image_.set_shape(params.OUT_HEIGHT*params.OUT_WIDTH*3)
    image_ = tf.reshape(image_, (params.OUT_HEIGHT,params.OUT_WIDTH,3))
    image = tf.cast(image_, tf.float32)/255.0
    
    
    name = features['name_raw']
    
    return image, image_, name



def inputs_trainval(file_list, batch_size, shuffle_flag, record_main_path, num_threads, augment=False, reduce_size=False, whiten=False, grayscale=False):

    filename = []
    f = open(file_list, 'r')
    for line in f:
        img_path = line[:-1]
        img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
#        record_path = '/'.join(img_path.split('/')[:-2]) + '/tfrecords_w/'
        record_path = record_main_path + img_name + '.' + params.DATA_EXT
        filename.append(record_path)

    N = len(filename)
    
    filename_queue = tf.train.string_input_producer(filename)

    # Even when reading in multiple threads, share the filename queue.
    image, image_, label, image_name = _read_and_decode_trainval(filename_queue)
    
    if reduce_size:
        image = tf.image.resize_images(image, tf.constant([params.OUT_HEIGHT, params.OUT_WIDTH]))
    
    if augment:
        image, image_, label, image_name = data_augment(image, image_, label, image_name)
        
    # do whitening
    if whiten:
        image = do_whitening(image)
    else:
        image = 2 * (image - 0.5)
        
    # convert to grayscale
    if grayscale:
        image = _to_grayscale(image)
        
    
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if shuffle_flag:
        images, images_, labels, image_names = tf.train.shuffle_batch(
            [image, image_, label, image_name], batch_size=batch_size, num_threads=num_threads,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    else:
        images, images_, labels, image_names = tf.train.batch(
            [image, image_, label, image_name], batch_size=batch_size, num_threads=num_threads,
            capacity=1000 + 3 * batch_size)

    return images, images_, labels, image_names, N
    

def inputs_test(file_list, batch_size, shuffle_flag):

#    filename = []
#    f = open(file_list, 'r')
#    for line in f:
#        img_name = line[:-1].split('/')[-1].split('.')[0]
#        filename.append(params.TEST_RECORDS_DATA_DIR + img_name + "." + params.DATA_EXT)
    
    filename = []
    f = open(file_list, 'r')
    for line in f:
        img_path = line[:-1]
        img_name = img_path.split('/')[-1].split('.')[0]
#        record_path = '/'.join(img_path.split('/')[:-2]) + '/tfrecords/' + img_name + '.' + params.DATA_EXT
        record_path = params.TEST_RECORDS_DATA_DIR + img_name + '.' + params.DATA_EXT
        filename.append(record_path)
    
    N = len(filename)

    filename_queue = tf.train.string_input_producer(filename)

    # Even when reading in multiple threads, share the filename queue.
    image, image_, image_name = _read_and_decode_test(filename_queue)
    
    # do whitening
    image = do_whitening(image)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if shuffle_flag:
        images, images_, image_names = tf.train.shuffle_batch(
            [image, image_, image_name], batch_size=batch_size, num_threads=params.NUM_THREADS_FOR_INPUT,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    else:
        images, images_, image_names = tf.train.batch(
            [image, image_, image_name], batch_size=batch_size, num_threads=params.NUM_THREADS_FOR_INPUT,
            capacity=1000 + 3 * batch_size)

    return images, images_, image_names, N


def _resize_and_pad(image):
    
    h = float(image.shape[0])
    w = float(image.shape[1])
    r_i = w / h
    
    if r_i>R_SZ:
        K = W_SZ / w
        w_ = W_SZ
        h_ = int(K * h)
        image = misc.imresize(image, K)
        pad_h = int((H_SZ - h_) / 2.0)
        pad_up = np.tile(np.expand_dims(image[0,:,:], axis=0), (pad_h, 1, 1))
        pad_down = np.tile(np.expand_dims(image[-1,:,:], axis=0), (pad_h, 1, 1))
#        pad_up = ndimage.gaussian_filter(pad_up, sigma=(3, 3, 0), order=0)
#        pad_down = ndimage.gaussian_filter(pad_down, sigma=(3, 3, 0), order=0)        
        image = np.concatenate((pad_up, image, pad_down), axis=0)
    else:
        K = H_SZ / h
        h_ = H_SZ
        w_ = K * w
        image = misc.imresize(image, K)
        pad_w = int((W_SZ - w_) / 2.0)
        pad_left = np.tile(np.expand_dims(image[:,0,:], axis=1), (1, pad_w, 1))
        pad_right = np.tile(np.expand_dims(image[:,-1,:], axis=1), (1, pad_w, 1))
#        pad_left = ndimage.gaussian_filter(pad_left, sigma=(3, 3, 0), order=0)
#        pad_right = ndimage.gaussian_filter(pad_right, sigma=(3, 3, 0), order=0)        
        image = np.concatenate((pad_left, image, pad_right), axis=1)

    image = misc.imresize(image, (H_SZ, W_SZ))
    
    return image
    

def _stretch(image):
    
    H_SZ = params.OUT_HEIGHT
    W_SZ = params.OUT_WIDTH

    image = misc.imresize(image, (H_SZ, W_SZ))
    
    K_x =   W_SZ / image.shape[1]
    K_y =   H_SZ / image.shape[0]
    offset_x = 0.0
    offset_y = 0.0
    
    transf_params = np.array((K_x, K_y, offset_x, offset_y), dtype=np.float32)
    
    return image, transf_params


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


#def convert_to_record(name, image, records_dir):
#    """Create a data record."""
#
##    if not os.path.exists(records_dir):
##        os.makedirs(records_dir)
#
#    filename = os.path.join(records_dir, name + '.tfrecords')
##    print('Writing', filename)
#
#    writer = tf.python_io.TFRecordWriter(filename)
#  
#    image_raw = image.tostring()
#    name_raw = name
#    example = tf.train.Example(features=tf.train.Features(feature={
#        'image_raw': _bytes_feature(image_raw),
#        'name_raw': _bytes_feature(name_raw)
#        }))        
#    writer.write(example.SerializeToString())
#    writer.close()
#    
#    return

def convert_to_record(file_name, image_name, image):

    writer = tf.python_io.TFRecordWriter(file_name)
    
    image_raw = image.tostring()
    name_raw = image_name
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'name_raw': _bytes_feature(name_raw)
        }))        
    writer.write(example.SerializeToString())
    writer.close()
    return



def inputs_test_quick(input_dir, processed_images_dir, batch_size, shuffle_flag, whiten=True):
    
    # create subdir and fill it with cropped-resized images
    record_list = []
    image_list = next(os.walk(input_dir))[2]
    
    N = len(image_list)
    for image_nm in image_list:
        image_in = misc.imread(os.path.join(input_dir, image_nm))
#        image_in = _resize_and_pad(image_in)
        image_in, _ = _stretch(image_in)
        record_name = os.path.join(processed_images_dir, '.'.join(image_nm.split('.')[:-1]) + '.' + params.DATA_EXT)
        if not os.path.exists(record_name):
            convert_to_record(record_name, image_nm, image_in)
            misc.imsave(os.path.join(processed_images_dir, image_nm), image_in)
        record_list.append(record_name)
    
    # make queue
    filename_queue = tf.train.string_input_producer(record_list)
    
    # write reader
    image, image_, name = _read_and_decode_test(filename_queue)
    
    
    # data whitening
    if whiten:
        image = do_whitening(image)
    else:
        image -= 0.5
    # make batches
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if shuffle_flag:
        images, images_, names = tf.train.shuffle_batch(
            [image, image_, name], batch_size=batch_size, num_threads=params.NUM_THREADS_FOR_INPUT,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    else:
        images, images_, names = tf.train.batch(
            [image, image_, name], batch_size=batch_size, num_threads=params.NUM_THREADS_FOR_INPUT,
            capacity=1000 + 3 * batch_size)
    
    return images, images_, names, N




#file_list = params.TRAINING_IMAGES_LIST
#batch_size = 2
#shuffle_flag = True
#images, images_, mapsB, mapsP, mapsX, labels, image_names = inputs_trainval(file_list, batch_size, shuffle_flag)

