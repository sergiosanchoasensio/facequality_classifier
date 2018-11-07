import atexit
import tensorflow as tf
import os
from subprocess import Popen
import signal
import time
import arch as arch
import params
import tools
import train_core
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from augmentations import data_augment
import tensorflow.contrib.slim as slim
import cv2


def start_tboard():
    os.system('fuser 6006/tcp -k')
    cmd_str = 'tensorboard --logdir ' + params.TRAINING_MODEL_DATA_DIR
    proc = Popen([cmd_str], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    global child_pid
    child_pid = proc.pid
    def kill_child():
        if child_pid is None: pass
        else: os.kill(child_pid, signal.SIGTERM)
    atexit.register(kill_child)


def read_and_decode(tfrecords_list, is_train, global_step):
    filenames = [l.replace('\n', '') for l in open(tfrecords_list)]
    
    if is_train:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=True, seed=1)
    else:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False, seed=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    
    crop_ = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    height = features['height']
    width = features['width']
    map_shape = tf.stack([1, height, width, 3])
    crop = tf.reshape(crop_, map_shape)
    
    crop_ = tf.image.resize_bilinear(crop, [params.OUT_HEIGHT, params.OUT_WIDTH], align_corners=True)
    crop = tf.squeeze(crop_)
    crop = crop[..., ::-1]
    
    crop /= 255.0
#    map_shape = tf.stack([params.OUT_HEIGHT, params.OUT_WIDTH, 3])
#    crop = tf.reshape(crop_, map_shape)
    if is_train and params.AUGMENT:
        crop = data_augment(crop)
    crop = tf.image.per_image_standardization(crop)
    crop = tf.expand_dims(crop, 0)

    label = tf.one_hot(features['label'], len(params.LABELS))
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [len(params.LABELS)])

    if is_train:
        crops, labels = tf.train.shuffle_batch([crop, label],
                                               batch_size=params.BATCH_SIZE,
                                               capacity = 1000 + 3 * params.BATCH_SIZE,
                                               num_threads=params.NUM_THREADS_FOR_INPUT,
                                               min_after_dequeue = 1000)
        crops = tf.reshape(crops, [params.BATCH_SIZE, params.OUT_HEIGHT, params.OUT_WIDTH, 3])
    else:
        crops, labels = tf.train.batch([crop, label],
                                       batch_size=1,
                                       capacity=1000 + 3 * 1,
                                       allow_smaller_final_batch=True)
        crops = tf.reshape(crops, [1, params.OUT_HEIGHT, params.OUT_WIDTH, 3])

    return crops, labels


def run(restore_path=None, show_images=False):
    with tf.device(params.DEVICE['tf_id']):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)

            crops = tf.placeholder(dtype=tf.float32, shape=[params.BATCH_SIZE, params.OUT_HEIGHT, params.OUT_WIDTH, 3])
            labels = tf.placeholder(dtype=tf.float32, shape=[params.BATCH_SIZE, len(params.LABELS)])

            preds_out, feat_out, weights, image_out, labels_out = arch.inference(crops, labels, True)
            loss = train_core.loss(preds_out, labels_out, feat_out, weights)
            train_op = train_core.train(loss, global_step, params.N_BOX_TR)
    
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            merged_summary_op_gen = tf.summary.merge(tf.get_collection('summaries'))

            summary_writer = tf.summary.FileWriter(params.LOGS_PATH, graph=tf.get_default_graph())

            config = tf.ConfigProto(allow_soft_placement=True)

            if restore_path is not None:
                variables = slim.get_variables_to_restore()
                variables_to_restore = [v for v in variables if (v.name.split('/')[0] not in params.NEW_VARIABLES) and ('ExponentialMovingAverage' not in v.name) and ('Adam' not in v.name)]
                loader = tf.train.Saver(variables_to_restore)

            saver = tf.train.Saver(max_to_keep=5000)

            with tf.Session(config=config) as sess:

                sess.run(init_op)

                if restore_path is not None:
                    loader.restore(sess, restore_path)
                    print(restore_path + ' loaded.')

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
    
                evals_dir = params.TRAINING_MODEL_DATA_DIR + 'evals/'
                if not os.path.exists(evals_dir):
                    os.makedirs(evals_dir)

                for n in xrange(params.MAX_STEPS):

                    cr, lab = tools.get_batch(params.tr_files, params.tr_labels, params.AUGMENT)
                    
                    if show_images:
                        from scipy.misc import imshow
                        for j in range(params.BATCH_SIZE):
                            im = cr[j]
                            print lab[j]
                            imshow(im)
#                    
                    t_s, loss_tr, im_out, lo = sess.run([train_op, loss, image_out, preds_out], feed_dict={crops: cr, labels: lab})

                    if n % 100 == 0:
                        print str(n) + ':', loss_tr
                        summary_str_gen = sess.run(merged_summary_op_gen, feed_dict={is_train: True, is_val_set: True})
                        summary_writer.add_summary(summary_str_gen, n)

                    if n == 1000:
                        tools.copy_arch_n_params(params.ARCH_PARAMS_DIR)                        

                    if (n % 1000 == 0 and n != 0) or (n == 1000):
                        
                        print 'Saving model...'
                        with tf.device('/cpu:0'):
                            saver.save(sess, params.TRAINING_MODEL_DATA_DIR + 'mdl_ckpt', global_step=n)
                            
                        print 'Evaluating...'
                        for val_set in [True, False]:
                            confu = np.zeros([len(params.LABELS), len(params.LABELS)], dtype=np.int32)

                            for j in range(params.N_BOX_VAL):
                                    cs, lo, po = sess.run([crops, labels_out, preds_out], feed_dict={is_train: False, is_val_set: val_set})
                                    po = sess.run(preds_out, feed_dict={crops: cr_, labels: lab_})

                                    label = np.argmax(lo)
                                    pred = np.argmax(po)
                                    confu[label, pred] += 1
                            s = 'validation' if val_set else 'train_small'
                            orig_stdout = sys.stdout
                            tools.save_metrics(evals_dir + 'log_' + s + '.txt', confu, orig_stdout, n)
                        tools.compare_val_ts(evals_dir)

                coord.request_stop()
                coord.join(threads)
                sess.close()


if __name__ == '__main__':
    start_tboard()
    run(show_images=True)




