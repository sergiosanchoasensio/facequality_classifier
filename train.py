import atexit
import tensorflow as tf
import os
from subprocess import Popen
import signal
import arch as arch
import params
import tools
import train_core
import numpy as np
import sys
import tensorflow.contrib.slim as slim


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


def run(restore_path=None, show_images=False):
    with tf.device(params.DEVICE['tf_id']):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)

            crops = tf.placeholder(dtype=tf.float32, shape=[None, params.OUT_HEIGHT, params.OUT_WIDTH, 3])  # dimension 0 = None makes the placeholder flexible for batch size
            labels = tf.placeholder(dtype=tf.float32, shape=[None, len(params.LABELS)])

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

                    cr, lab = tools.get_batch(params.tr_files, params.tr_labels, is_train=True)
                    
                    if show_images:
                        from scipy.misc import imshow
                        for j in range(params.BATCH_SIZE):
                            im = cr[j]
                            print lab[j]
                            imshow(im)

                    t_s, loss_tr, im_out, po = sess.run([train_op, loss, image_out, preds_out], feed_dict={crops: cr, labels: lab})

                    if n % 100 == 0:
                        print str(n) + ':', loss_tr
                        summary_str_gen = sess.run(merged_summary_op_gen, feed_dict={crops: cr, labels: lab})
                        summary_writer.add_summary(summary_str_gen, n)

                    if n == 1000:
                        tools.copy_arch_n_params(params.ARCH_PARAMS_DIR)                        

                    if (n % params.EVA_FREQ == 0 and n != 0) or (n == 1000):
                        
                        print 'Saving model...'
                        with tf.device('/cpu:0'):
                            saver.save(sess, params.TRAINING_MODEL_DATA_DIR + 'mdl_ckpt', global_step=n)
                            
                        print 'Evaluating...'
                        for val_set in [True, False]:
                            confu = np.zeros([len(params.LABELS), len(params.LABELS)], dtype=np.int32)
                            if val_set:
                                N = params.N_BOX_VAL
                                files = params.val_files
                                labs = params.val_labels
                                s = 'validation'
                            else:
                                N = params.N_BOX_TS
                                files = params.ts_files
                                labs = params.ts_labels
                                s = 'train_small'
                            for j in range(N):
                                cr_, lab_ = tools.get_batch(files, labs, is_train=False, id_file=j)
                                po = sess.run(preds_out, feed_dict={crops: cr_, labels: lab_})
                                label = np.argmax(lab_)
                                pred = np.argmax(po)
                                confu[label, pred] += 1
                            orig_stdout = sys.stdout
                            tools.save_metrics(evals_dir + 'log_' + s + '.txt', confu, orig_stdout, n)
                        tools.compare_val_ts(evals_dir)

                coord.request_stop()
                coord.join(threads)
                sess.close()


if __name__ == '__main__':
    start_tboard()
    run(show_images=False)




