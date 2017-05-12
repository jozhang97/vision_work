import tensorflow as tf
import math
device_name = "/gpu:0"

def variable_summaries(name, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.device(device_name):
    with tf.name_scope(name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def variable_summaries_map(mapp):
  with tf.device(device_name):
    for name, vals in mapp.items():
        variable_summaries("Weight_" + name, vals)

def weight_variable(shape, stddev=0.1):
  with tf.device(device_name):
    # If convolutional layer, make the stddev follow this
    if len(shape) == 4: 
        k_1, k_2, in_chan, out_chan = shape
        n_l = k_2 * k_1 * in_chan
        stddev = math.sqrt(2/n_l)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.0):
  with tf.device(device_name):
    constant = 0.0 # per "delving Deep into rectifers"
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=2):
  with tf.device(device_name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2x2(x, stride = 2):
  with tf.device(device_name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1,stride, stride, 1], padding='SAME')

def max_pool_3x3(x, stride=2):
  with tf.device(device_name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                      strides=[1, stride, stride, 1], padding='SAME')

def avg_pool_3x3(x, stride=2):
  with tf.device(device_name):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                      strides=[1, stride, stride, 1], padding='SAME')

def avg_pool_2x2(x, stride=2):
  with tf.device(device_name):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, stride, stride, 1], padding='SAME')

