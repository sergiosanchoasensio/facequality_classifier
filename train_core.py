import numpy as np
import tensorflow as tf
import params


def cross_entropy(labels, predictions, class_weights=1):
    return - tf.reduce_sum(labels * tf.log(predictions + np.finfo(np.float32).eps) * class_weights, 1)


def sym_cross_entropy(labels, predictions, class_weights=1.0):
    return - tf.reduce_sum(labels * tf.log(predictions + np.finfo(np.float32).eps) * class_weights, 1) - tf.reduce_sum((1.0 - labels) * tf.log((1.0 - predictions) + np.finfo(np.float32).eps) * class_weights, 1)


def loss(labels, preds, labels_blur, preds_blur, features_out, w_to_reg=[]):
    """L2 loss

    Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """

    # total loss per crop
    loss_orientation = sym_cross_entropy(labels, preds)
    loss_orientation = tf.reduce_mean(loss_orientation, 0)
    loss_orientation = tf.identity(loss_orientation, name='loss_ce_orientation')
    try:
        loss_blur = sym_cross_entropy(labels_blur, preds_blur)
    except:
        import pdb;pdb.set_trace()
    loss_blur = tf.reduce_mean(loss_blur, 0)
    loss_blur = tf.identity(loss_blur, name='loss_ce_blur')
    loss = params.LAMBDA_ORIENTATION * loss_orientation + params.LAMBDA_BLUR * loss_blur
    tf.add_to_collection('losses', loss)

    # feature reciprocal norm
    if params.LAMBDAfrn != 0.0:

        lab_id = tf.argmax(labels, axis=1)
        pre_id = tf.argmax(preds, axis=1)
        h_i = tf.cast(tf.equal(lab_id, pre_id), dtype=tf.float32)
        n_i = tf.reduce_mean(preds ** 2, axis=1)
        frn_loss = params.LAMBDAfrn * tf.reduce_mean(h_i / (n_i + np.finfo(np.float32).eps))
        frn_loss = tf.identity(frn_loss, name='loss_featureReciprocalNorm')

        lab_id_blur = tf.argmax(labels_blur, axis=1)
        pre_id_blur = tf.argmax(preds_blur, axis=1)
        h_i_blur = tf.cast(tf.equal(lab_id_blur, pre_id_blur), dtype=tf.float32)
        n_i_blur = tf.reduce_mean(preds_blur ** 2, axis=1)
        frn_loss_blur = params.LAMBDAfrn * tf.reduce_mean(h_i_blur / (n_i_blur + np.finfo(np.float32).eps))
        frn_loss_blur = tf.identity(frn_loss_blur, name='loss_featureReciprocalNorm_blur')

        tf.add_to_collection('losses', 0.5 * frn_loss + 0.5 * frn_loss_blur)

    return tf.add_n(tf.get_collection('losses'), name='loss_total')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in the model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')    
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.

    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        #    tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, n_examples):
  """Train the model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # load weight decay loss if any
  wd_loss_list = tf.get_collection('weight_losses')
  if wd_loss_list != []:
      total_weight_loss = tf.add_n(tf.get_collection('weight_losses'), name='weight_loss_total')
      tf.summary.scalar('loss_weight_decay', total_weight_loss)
  
  # Variables that affect learning rate.
  num_batches_per_epoch = n_examples / params.BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * params.NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(params.INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  params.LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  lr = params.INITIAL_LEARNING_RATE
  tf.summary.scalar('learningRate', lr)

  # Generate moving averages of all losses and associated summaries.
#  loss_names = params.LOSS_NAMES
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
#    opt = tf.train.RMSPropOptimizer(lr, momentum=params.OPT_MOMENTUM)
    opt = tf.train.AdamOptimizer(lr, beta1=params.ADAM_B1)
    opt_wd = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    if wd_loss_list != []:
        grads_wd = opt_wd.compute_gradients(total_weight_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  if wd_loss_list != []:
      apply_gradient_wd_op = opt_wd.apply_gradients(grads_wd)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  if wd_loss_list != []:
    for grad, var in grads_wd:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients_wd', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(params.MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  if wd_loss_list != []:
      with tf.control_dependencies([apply_gradient_op, apply_gradient_wd_op, variables_averages_op]):
#          train_op = tf.no_op(name='train')
          train_op = tf.group(apply_gradient_op, apply_gradient_wd_op, name='train')
  else:
      with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
          train_op = tf.no_op(name='train')
      
  return train_op

