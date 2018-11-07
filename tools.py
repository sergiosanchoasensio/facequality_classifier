import cv2
import tensorflow as tf
import os
import shutil
import params
import numpy as np
import sys
import matplotlib.pyplot as plt
import ast
from augmentations_np import augment_tr, blur, motion_blur
from random import randint


np.random.seed(0)


def hflip_imgs(image):
    flip = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.bool)
    image_flip = tf.reverse(image, [1])
    return tf.cond(flip, lambda: image_flip, lambda: image)


def arch_in_train():
    imp = 'import '
    with open('train.py') as fp:
        for i, line in enumerate(fp):
            if 'arch' in line:
                if '#' in line:
                    if line.index('#') != 0:
                        end = ' as '
                        return line[len(imp):line.index(end)]
                else:
                    end = ' as '
                    return line[len(imp):line.index(end)]


def copy_arch_n_params(dest_dir):
    arch = arch_in_train() + '.py'
    prms = 'params.py'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.copy(arch, dest_dir)
    shutil.copy(prms, dest_dir)
    return None


def eval_metrics(confu):
    print 'Confusion matrix:\n', confu

    N = np.sum(confu)
    N_correct = np.sum(confu.diagonal())
    Acc = np.divide(np.float(N_correct), N)
    print 'Accuracy:', Acc

    P = []
    i = 0
    for col in np.transpose(confu):
        TP = col[i]
        TP_FP = np.sum(col)
        prec = np.divide(np.float(TP), TP_FP + params.EPSILON)
        P += [prec]
        i += 1
    avg_prec = np.mean(P)
    print 'Average precision:', avg_prec
    print 'Precision per color:', P

    S = []
    j = 0
    for row in confu:
        TP = row[j]
        TP_FP = np.sum(row)
        sen = np.divide(np.float(TP), TP_FP + params.EPSILON)
        S += [sen]
        j += 1
    avg_sen = np.mean(S)
    print 'Average sensitivity:', avg_sen
    print 'Sensitivity per color:', S

    F1 = []
    for k in range(confu.shape[0]):
        f1 = 2 * (P[k] * S[k]) / (P[k] + S[k] + params.EPSILON)
        F1 += [f1]
    avg_F1 = np.mean(F1)
    print 'Average F1:', avg_F1
    print 'F1 per color:', F1


def metric_list(metric, lines, steps=False):
    content = []
    for l in lines:
        l_ = l.strip().replace('\n', '')
        if metric in l_:
            if steps:
                content += [np.float(l_[len('Evaluation at step : ')-2:-1])]
            else:
                try:
                    content += [np.float(l_[len(metric) + 2:])]
                except:
                    content += [ast.literal_eval(l_[len(metric) + 2:])]
    return content


def compare_val_ts(evals_dir):
    metrics = ['Accuracy', 'Average precision', 'Average sensitivity', 'Average F1']
    for m in metrics:
        cols = []
        for y in ['orientation', 'blurry']:
            for path in [evals_dir + 'log_' + y + '_train_small.txt', evals_dir + 'log_' + y + '_validation.txt']:
                lines = open(path).readlines()
                cols += [metric_list(m, lines)]
                steps = metric_list('step', lines, True)
            fig = plt.figure()
            fig.suptitle(m)
            line_up, = plt.plot(steps, cols[0], '--', label='training', color='k')
            line_down, = plt.plot(steps, cols[1], label='validation', color='k')
            plt.legend(handles=[line_up, line_down])
            plt.savefig(evals_dir + m + '_' + y + '.jpg')
            plt.close()


def save_metrics(path, confu, orig_stdout, n):
    f = open(path, 'a')
    sys.stdout = f
    print '\nEvaluation at step', str(n) + ':'
    eval_metrics(confu)
    sys.stdout = orig_stdout
    f.close()


def gamma_correct(image):
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, params.GAMMA_PROB)
    p_order_more_less = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred_more_less = tf.less(p_order_more_less, 0.5)
    image = tf.cond(pred, lambda: tf.cond(pred_more_less, lambda: image ** 0.75, lambda: image ** 1.5), lambda: image)

    return image


def _change_illuminant_blue(image):
    i0_r = tf.slice(image, [0, 0, 0], [-1, -1, 1])
    i0_g = tf.slice(image, [0, 0, 1], [-1, -1, 1])
    i0_b = tf.slice(image, [0, 0, 2], [-1, -1, 1]) * 0.8 + 0.2

    if params.USE_TEMPORAL_INFO:
        i1_r = tf.slice(image, [0, 0, 3], [-1, -1, 1])
        i1_g = tf.slice(image, [0, 0, 4], [-1, -1, 1])
        i1_b = tf.slice(image, [0, 0, 5], [-1, -1, 1]) * 0.8 + 0.2
        i2_r = tf.slice(image, [0, 0, 6], [-1, -1, 1])
        i2_g = tf.slice(image, [0, 0, 7], [-1, -1, 1])
        i2_b = tf.slice(image, [0, 0, 8], [-1, -1, 1]) * 0.8 + 0.2
        image = tf.concat(axis=2, values=[i0_r, i0_g, i0_b, i1_r, i1_g, i1_b, i2_r, i2_g, i2_b])
    else:
        image = tf.concat(axis=2, values=[i0_r, i0_g, i0_b])

    return image


def _change_illuminant_orange(image):
    i0_r = tf.slice(image, [0, 0, 0], [-1, -1, 1])
    i0_g = (1.0 / 3.0) * tf.slice(image, [0, 0, 1], [-1, -1, 1]) * 0.5412 + (2.0 / 3.0) * tf.slice(image, [0, 0, 1],
                                                                                                   [-1, -1, 1])
    i0_b = (1.0 / 3.0) * tf.slice(image, [0, 0, 2], [-1, -1, 1]) * 0.1255 + (2.0 / 3.0) * tf.slice(image, [0, 0, 2],
                                                                                                   [-1, -1, 1])

    if params.USE_TEMPORAL_INFO:
        i1_r = tf.slice(image, [0, 0, 3], [-1, -1, 1])
        i1_g = (1.0 / 3.0) * tf.slice(image, [0, 0, 4], [-1, -1, 1]) * 0.5412 + (2.0 / 3.0) * tf.slice(image, [0, 0, 4],
                                                                                                       [-1, -1, 1])
        i1_b = (1.0 / 3.0) * tf.slice(image, [0, 0, 5], [-1, -1, 1]) * 0.1255 + (2.0 / 3.0) * tf.slice(image, [0, 0, 5],
                                                                                                       [-1, -1, 1])
        i2_r = tf.slice(image, [0, 0, 6], [-1, -1, 1])
        i2_g = (1.0 / 3.0) * tf.slice(image, [0, 0, 7], [-1, -1, 1]) * 0.5412 + (2.0 / 3.0) * tf.slice(image, [0, 0, 7],
                                                                                                       [-1, -1, 1])
        i2_b = (1.0 / 3.0) * tf.slice(image, [0, 0, 8], [-1, -1, 1]) * 0.1255 + (2.0 / 3.0) * tf.slice(image, [0, 0, 8],
                                                                                                       [-1, -1, 1])
        image = tf.concat(axis=2, values=[i0_r, i0_g, i0_b, i1_r, i1_g, i1_b, i2_r, i2_g, i2_b])
    else:
        image = tf.concat(axis=2, values=[i0_r, i0_g, i0_b])

    return image


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


def add_chroma_noise(image):
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, params.NOISE_PROB)
    image = tf.cond(pred, lambda: _add_chroma_noise(image), lambda: image)

    return image


def _change_illuminant(image):
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)

    image = tf.cond(pred, lambda: _change_illuminant_blue(image), lambda: _change_illuminant_orange(image))

    return image


def change_illuminant(image):
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, params.ILLU_PROB)
    image = tf.cond(pred, lambda: _change_illuminant(image), lambda: image)

    return image


def def_init_sttdev(activation_function, input_n_feat, set_max=None):

    if activation_function=='linear':
        stddev = np.sqrt(1.0 / input_n_feat)
    elif activation_function=='relu':
        stddev = np.sqrt(2.0 / input_n_feat)
    elif activation_function=='leaky_relu':
        stddev = np.sqrt(1.98 / input_n_feat)
    elif activation_function=='tanh':
        stddev = np.sqrt(1.0 / input_n_feat)
    elif activation_function=='sigmoid':
        stddev = 3.6 * np.sqrt(1.0 / input_n_feat)
    elif activation_function=='softplus':
        stddev = 1.537 / np.sqrt(input_n_feat)
    else:
        raise AssertionError('activation function not found')

    if set_max is not None:
        stddev = np.minimum(stddev, set_max)

    return stddev


def get_batch(file_list, lab_list, is_train, id_file=None):
    if is_train:
        idx = np.random.choice(len(file_list), params.BATCH_SIZE)
        fnames = [f for j, f in enumerate(file_list) if j in idx]
        labs = np.array([f for j, f in enumerate(lab_list) if j in idx])
    else:
        fnames = [file_list[id_file]]
        labs = np.array([lab_list[id_file]])
    crops = []
    labs_blurry = []
    for m, f in enumerate(fnames):
        crop = cv2.imread(f)[..., ::-1]
        crop = cv2.resize(crop, (params.OUT_WIDTH, params.OUT_HEIGHT))
        crop = crop.astype(np.float32)
        kernel = randint(5, 9)
        kernel_m = randint(7, 25)
        if is_train:
            if m % 2 == 0:  # for half of the batch, we do blurring and change the label to not ok
                which_blur = randint(0, 1)
                if which_blur:
                    crop = blur(crop, kernel)
                else:
                    crop = motion_blur(crop, kernel_m)
                labs_blurry += [[0.0, 1.0]]
            else:
                labs_blurry += [[1.0, 0.0]]
            if params.AUGMENT:
                crop = augment_tr(crop)
        else:
            blurring = id_file % 2 == 0  # one eval sample over 2
            if blurring:
                which_blur = randint(0, 1)
                if which_blur:
                    crop = blur(crop, kernel)
                else:
                    crop = motion_blur(crop, kernel_m)
                labs_blurry = [[0.0, 1.0]]
            else:
                labs_blurry = [[1.0, 0.0]]
        crop /= 255.0
        crops += [crop]
    crops = np.array(crops)
    labs_blurry = np.array(labs_blurry)
    return crops, labs, labs_blurry
