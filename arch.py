from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import params
import numpy as np


def leaky_relu(x, slope, name):
    out = tf.maximum(x, slope * x)

    return tf.identity(out, name=name)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % params.TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
	trainable=trainable)
    if wd is not None:
    	weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
	tf.add_to_collection('weight_losses', weight_decay)
    return var


def conv_nin_res(input_data, shape, init_stddev, bias_init, name, add_residual=False):
    # conv
    kernel = _variable_with_weight_decay('weights_' + name,
                                         shape=shape,
                                         # set shape=[5, 5, 9, 64] with image sequence input, [5, 5, 3, 64] without
                                         stddev=def_init_sttdev('leaky_relu', input_n_feat=shape[0] * shape[1] * int(shape[2]), set_max=0.01),
                                         wd=params.L2_WEIGHT_DECAY)
    conv_ = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding='SAME')
    bias = _variable_on_cpu('biases_' + name, shape[3], tf.constant_initializer(bias_init))
    conv_ = tf.nn.bias_add(conv_, bias)
    #    conv = tf.nn.relu(conv_, name=name)
    conv = leaky_relu(conv_, slope=0.1, name=name)

    # conv nin
    kernel_nin = _variable_with_weight_decay('weights_nin_' + name,
                                             shape=[1, 1, shape[3], shape[3]],
                                             # set shape=[5, 5, 9, 64] with image sequence input, [5, 5, 3, 64] without
                                             stddev=def_init_sttdev('leaky_relu', input_n_feat=int(shape[2]), set_max=0.01),
                                             wd=params.L2_WEIGHT_DECAY)
    conv_nin = tf.nn.conv2d(conv, kernel_nin, [1, 1, 1, 1], padding='SAME')
    bias_nin = _variable_on_cpu('biases_nin_' + name, shape[3], tf.constant_initializer(bias_init))
    conv_nin = tf.nn.bias_add(conv_nin, bias_nin)
    #    conv_nin = tf.nn.relu(conv_nin, name=name)
    conv_nin = leaky_relu(conv_nin, slope=0.1, name=name)

    if add_residual:
        conv_nin += conv

    return conv_nin, kernel, bias, kernel_nin, bias_nin


def nin_block(input_data, nin_shape, scope_name):
    conv_nin, kernel, bias, kernel_nin, bias_nin = conv_nin_res(input_data,
                                                                shape=nin_shape,
                                                                init_stddev=5e-2,
                                                                bias_init=0.1,
                                                                name=scope_name,
                                                                add_residual=True)

    output_data = conv_nin

    return output_data, kernel, bias, kernel_nin, bias_nin


def def_init_sttdev(activation_function, input_n_feat, set_max=None):

    if activation_function == 'linear':
        stddev = np.sqrt(1.0 / input_n_feat)
    elif activation_function == 'relu':
        stddev = np.sqrt(2.0 / input_n_feat)
    elif activation_function == 'leaky_relu':
        stddev = np.sqrt(1.98 / input_n_feat)
    elif activation_function == 'tanh':
        stddev = np.sqrt(1.0 / input_n_feat)
    elif activation_function == 'sigmoid':
        stddev = 3.6 * np.sqrt(1.0 / input_n_feat)
    elif activation_function == 'softplus':
        stddev = 1.537 / np.sqrt(input_n_feat)
    else:
        raise AssertionError('activation function not found')

    if set_max is not None:
        stddev = np.minimum(stddev, set_max)

    return stddev


def inference(images, labels, labels_blur, trainval_flag):
    """Build the model

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().


    weights_data = dict()

    n_feat = {}

    n_feat['conv1'] = 16#32
    n_feat['conv2'] = 32#64
    n_feat['conv3'] = 32#96
    n_feat['conv4'] = 32#192
    n_feat['conv5'] = 64#192
    n_feat['conv6'] = 64#128
    n_feat['conv7'] = 64#128
    n_feat['fc8'] = 32#64
    #  n_feat['fc9'] = 64


    # layer1
    with tf.variable_scope('layer1') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(images, nin_shape=[5, 5, 3, n_feat['conv1']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out1_size = [params.OUT_HEIGHT / 2, params.OUT_WIDTH / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output1 = pool

    ######################################################################################################
    # layer2
    with tf.variable_scope('layer2') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output1, nin_shape=[3, 3, n_feat['conv1'], n_feat['conv2']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out2_size = [out1_size[0] / 2, out1_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output2 = pool

    ######################################################################################################
    # layer3
    with tf.variable_scope('layer3') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output2, nin_shape=[3, 3, n_feat['conv2'], n_feat['conv3']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out3_size = [out2_size[0] / 2, out2_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output3 = pool

    ######################################################################################################
    # layer4
    with tf.variable_scope('layer4') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output3, nin_shape=[3, 3, n_feat['conv3'], n_feat['conv4']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out4_size = [out3_size[0] / 2, out3_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output4 = pool

    ######################################################################################################
    # layer5
    with tf.variable_scope('layer5') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output4, nin_shape=[3, 3, n_feat['conv4'], n_feat['conv5']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out5_size = [out4_size[0] / 2, out4_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output5 = pool

    ######################################################################################################
    # layer6
    with tf.variable_scope('layer6') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output5, nin_shape=[3, 3, n_feat['conv5'], n_feat['conv6']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out6_size = [out5_size[0] / 2, out5_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output6 = pool

    ######################################################################################################
    # layer7
    with tf.variable_scope('layer7') as scope:
        # NiN
        output_nin, k_0, b_0, k_nin, b_nin = nin_block(output6, nin_shape=[3, 3, n_feat['conv6'], n_feat['conv7']],
                                                       scope_name=scope.name)

        # pooling
        pool = tf.nn.max_pool(output_nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=scope.name + '_pool')
        out7_size = [out6_size[0] / 2, out6_size[1] / 2]

        _activation_summary(output_nin)

        # store parameters
        weights_data[scope.name + '_k_0'] = k_0
        weights_data[scope.name + '_b_0'] = b_0
        weights_data[scope.name + '_k_nin'] = k_nin
        weights_data[scope.name + '_b_nin'] = b_nin

        output7 = pool

    ######################################################################################################
    # fc8
    with tf.variable_scope('fc8') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, n_feat['conv7'] * out7_size[0] * out7_size[1], n_feat['fc8']],
                                             stddev=def_init_sttdev('softplus', input_n_feat=int(output7.get_shape()[-1]), set_max=0.01),
                                             wd=params.L2_WEIGHT_DECAY)
        conv_reshape = tf.reshape(output7, [-1, 1, 1, int(output7.shape[1]) * int(output7.shape[2]) * int(output7.shape[3])])
        #    dim = reshape.get_shape()[1].value
        conv8 = tf.nn.conv2d(conv_reshape, kernel, [1, 1, 1, 1], padding='SAME')

        biases = _variable_on_cpu('biases', [n_feat['fc8']], tf.constant_initializer(0.0))
        fc8 = tf.nn.bias_add(conv8, biases)
        #    fc8 = tf.nn.tanh(fc8)
        fc8 = tf.nn.softplus(fc8)
        #    fc8 = tf.nn.sigmoid(fc8)
        fc8 = tf.squeeze(fc8, axis=[1, 2])
        _activation_summary(fc8)

        # add dropout
        if params.TOGGLE_DROPOUT and trainval_flag:
            fc8 = tf.nn.dropout(fc8, keep_prob=0.5)

        weights_data['w8'] = kernel
        weights_data['b8'] = biases

    with tf.variable_scope('outSoftmax') as scope:
        weights = _variable_with_weight_decay('weights', [n_feat['fc8'], len(params.LABELS)],
                                              stddev=def_init_sttdev('linear', input_n_feat=int(fc8.shape[-1]), set_max=0.01), wd=params.L2_WEIGHT_DECAY_SOFTMAX)
        biases = _variable_on_cpu('biases', [len(params.LABELS)], tf.constant_initializer(0.0))
        out_prob = tf.add(tf.matmul(fc8, weights), biases, name=scope.name)
        out_pred = tf.nn.softmax(out_prob)
        _activation_summary(out_pred)
        weights_data['w_SoftMax'] = weights
        weights_data['b_SoftMax'] = biases

    with tf.variable_scope('outSoftmax_blur') as scope:
        weights_blur = _variable_with_weight_decay('weights', [n_feat['fc8'], len(params.LABELS_BLUR)],
                                              stddev=def_init_sttdev('linear', input_n_feat=int(fc8.shape[-1]), set_max=0.01), wd=params.L2_WEIGHT_DECAY_SOFTMAX)
        biases_blur = _variable_on_cpu('biases', [len(params.LABELS)], tf.constant_initializer(0.0))
        out_prob_blur = tf.add(tf.matmul(fc8, weights), biases_blur, name=scope.name)
        out_pred_blur = tf.nn.softmax(out_prob_blur)
        _activation_summary(out_pred_blur)
        weights_data['w_SoftMax_blur'] = weights_blur
        weights_data['b_SoftMax_blur'] = biases_blur

    return out_pred, out_pred_blur, fc8, weights_data, images, labels, labels_blur