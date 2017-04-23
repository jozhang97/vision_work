import tensorflow as tf
import numpy as np
from data_generator import Cifar
cifar = Cifar()

device_name = "/gpu:0"
''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 0.1
learning_rate_placeholder = tf.placeholder(tf.float32)
tf.summary.scalar('learning_rate', learning_rate_placeholder)
learning_rate_decay = 0.1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
DELAYS_PER_EPOCH = 1
num_epochs_per_decay = 350

batch_size = 100 
reg_coeff = 0.00005
dropout_keep_prob = tf.placeholder(tf.float32)

''' HELPER FUNCTION '''
def variable_summaries(name, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
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
    for vals in mapp.values():
        variable_summaries(vals)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with tf.device(device_name):
    ''' DEFINE VARIABLES '''
    W = {
        "W_1": weight_variable([256, n_classes]),
        "W_2": weight_variable([20*20*64, 1024]), 
        "W_3": weight_variable([20*20*64, 1024]), 
        "W_4": weight_variable([5, 5, 64, 64]), 
        "W_5": weight_variable([5, 5, 3, 64]), 
        }
    b = {
        "b_1": bias_variable([n_classes]),
        "b_2": bias_variable([1024]),
        "b_3": bias_variable([512]),
        "b_4": bias_variable([64]),
        "b_5": bias_variable([64]),
        }
    variable_summaries_map(W)
    variable_summaries_map(b)

    x = tf.placeholder(tf.float32, [None, 3072])
    x_reshaped = tf.reshape(x, [-1, 32, 32, 3]) 

    y_5 = tf.nn.local_response_normalization(max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(x_reshaped, W["W_5"]), b["b_5"]))))

    y_4 = tf.nn.max_pool(tf.nn.local_response_normalization(tf.nn.relu(tf.nn.bias_ass(conv2d(y_5, W["W_4"]), b["b_4"]))))
    y_4 = tf.reshape(y_4, [-1, 20*20*64])

    y_3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_4, W["W_3"]), b["b_3"]))

    y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_3, W["W_2"]), b["b_2"]))
    y_2 = tf.nn.dropout(y_2, dropout_keep_prob)
    
    y_1 = tf.nn.bias_add(tf.matmul(y_2, W["W_1"]), b["b_1"]))
    y_true = tf.placeholder(tf.float32, [None, n_classes])


    ''' DEFINE LOSS FUNCTION '''
    cross_entropy_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
    tf.summary.scalar("cross_entropy", cross_entropy_2)
    loss = cross_entropy_2 
    tf.summary.scalar("loss", loss)


    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_1_eval, 1), tf.argmax(y_true, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


''' TRAIN '''
merged = tf.summary.merge_all()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)
sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('tensorboard_log_cifar/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_cifar/test')
sess.run(tf.global_variables_initializer())
for i in range(60000 * 10):
    if i % 100 == 0:
        summary, acc = sess.run([merged,accuracy], feed_dict={x: cifar.test_images, y_true: cifar.test_labels, dropout_keep_prob: 1, learning_rate_placeholder: learning_rate})
        test_writer.add_summary(summary, i)
        print(acc)
    else:
        batch_xs, batch_ys = cifar.train_next_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_true:batch_ys, dropout_keep_prob: 0.5, learning_rate_placeholder: learning_rate})
        train_writer.add_summary(summary, i)
    if i % 10000 == 0 and i > 0:
        learning_rate *= learning_rate_decay
