import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
cifar= input_data.read_data_sets("MNIST_data/", one_hot=True)

device_name = "/gpu:0"
''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 1e-4
learning_rate_placeholder = tf.placeholder(tf.float32)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('learning_rate_placeholder', learning_rate_placeholder)

learning_rate_decay = 0.1 
decays_per_epoch= 1/10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000 

batch_size = 100 
reg_coeff = 0.00005
epsilon = tf.Variable(0.000000000000001 * np.ones([n_classes]), dtype=tf.float32)
dropout_keep_prob = tf.placeholder(tf.float32)

''' HELPER FUNCTION '''
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
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
    W = {"W_1": weight_variable([1024, n_classes]),
        "W_2": weight_variable([7*7*64, 1024]), 
        "W_3": weight_variable([5, 5, 32, 64]),
        "W_4": weight_variable([5, 5, 1, 32]), 
        }
    b = {"b_1": bias_variable([n_classes]),
        "b_2": bias_variable([1024]),
        "b_3": bias_variable([64]),
        "b_4": bias_variable([32]),
        }
    variable_summaries_map(W)
    variable_summaries_map(b)

    x = tf.placeholder(tf.float32, [None, 784])
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1]) 

    y_4 = max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(x_reshaped, W["W_4"]), b["b_4"])))
    # y_4 = tf.nn.local_response_normalization(y_4)

    y_3 = max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(y_4, W["W_3"]), b["b_3"])))
    # y_3 = tf.nn.max_pool(tf.nn.local_response_normalization(tf.nn.relu(tf.nn.bias_ass(conv2d(y_4, W["W_3"]), b["b_3"]))))
    y_3 = tf.reshape(y_3, [-1, 7*7*64])

    y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_3, W["W_2"]), b["b_2"]))
    y_2 = tf.nn.dropout(y_2, dropout_keep_prob)
    y_1 = tf.nn.bias_add(tf.matmul(y_2, W["W_1"]) , b["b_1"])
    y_true = tf.placeholder(tf.float32, [None, n_classes])

    # # LOSS FUNCTIONS
    # with tf.variable_scope('weights_norm') as scope:
    #     weights_norm = tf.reduce_sum(
    #         input_tensor = WEIGHT_DECAY_FACTOR*tf.pack(
    #           [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
    #         ),
    #         name='weights_norm'
    # )
    # tf.summary.scalar('weights_norm', weights_norm)
    # tf.add_to_collection('losses', weights_norm)
    # ''' DEFINE LOSS FUNCTION '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
    tf.summary.scalar('cross_entropy', cross_entropy)
    # tf.add_to_collection('losses', cross_entropy)
    # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # reg_coeff = 0.01
    # regularization = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) + tf.nn.l2_loss(W_22) + tf.nn.l2_loss(b_22) + tf.nn.l2_loss(W_4) + tf.nn.l2_loss(b_4) + tf.nn.l2_loss(W_3) + tf.nn.l2_loss(b_3)
    # tf.summary.scalar("regularization", regularization)
    # loss = cross_entropy + reg_coeff * regularization
    loss = cross_entropy
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss', loss) 

    

    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_1, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('accuracy', accuracy)


''' TRAIN '''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
#sess = tf.Session()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
train_writer = tf.summary.FileWriter('tensorboard_log_mnist/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_mnist/test')
sess.run(init)
for i in range(20000):
    batch_xs, batch_ys = cifar.train.next_batch(batch_size)
    if i % 100 == 0 and i > 0:
        summary,acc= sess.run([merged, accuracy], feed_dict={x:cifar.test.images, y_true: cifar.test.labels, dropout_keep_prob: 1, learning_rate_placeholder: learning_rate})
        test_writer.add_summary(summary, i)
        print(acc)
    else:
        summary,_ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_true:batch_ys, dropout_keep_prob: 0.5, learning_rate_placeholder: learning_rate})
        train_writer.add_summary(summary, i)
    if i % 6000 == 0 and i > 0:
    #if i%NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * decays_per_epoch == 0:
        learning_rate *= learning_rate_decay 
