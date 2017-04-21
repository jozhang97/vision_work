import tensorflow as tf
import numpy as np
from data_generator import Cifar
cifar = Cifar()

''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 0.1
tf.summary.scalar("learning_rate", learning_rate)
learning_rate_decay = 0.1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
DELAYS_PER_EPOCH = 1
num_epochs_per_decay = 350
num_iter = 10000
batch_size = 100 
reg_coeff = 0.00005
epsilon = tf.Variable(0.000000000000001 * np.ones([n_classes]), dtype=tf.float32)
dropout_keep_prob = 0.75

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

''' DEFINE VARIABLES '''
x = tf.placeholder(tf.float32, [None, 3072])
x_reshaped = tf.reshape(x, [-1, 32, 32, 3]) 
W_4 = tf.Variable(tf.truncated_normal([5,5,3,64], mean=0, stddev=1.0))
variable_summaries("W_4", W_4)
b_4 = tf.Variable(tf.truncated_normal([64], mean=0, stddev=1.0))
conv_4 = tf.nn.conv2d(x_reshaped, W_4, strides=[1,1,1,1], padding="VALID")
relu_4 = tf.nn.relu(tf.nn.bias_add(conv_4, b_4))
pooled_4 = tf.nn.max_pool(relu_4, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")
y_4 = tf.nn.local_response_normalization(pooled_4)
W_3 = tf.Variable(tf.truncated_normal([5,5,64,64]), mean=0, stddev=1.0))
variable_summaries("W_3", W_3)
b_3 = tf.Variable(tf.truncated_normal([64], mean=0, stddev=1.0))
variable_summaries("b_3", b_3)
conv_3 = tf.nn.conv2d(y_4, W_3, strides=[1,1,1,1], padding="VALID")
conv_3 = tf.nn.local_response_normalization(conv_3)
relu_3 = tf.nn.relu(tf.nn.bias_add(conv_3, b_3))
pooled_3 = tf.nn.max_pool(relu_3, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")
y_3 = tf.reshape(pooled_3, [-1, 20*20*64])
x_22 = y_3
W_22 = tf.Variable(tf.truncated_normal([20*20*64,512], mean=0, stddev=1.0))
variable_summaries("W_22", W_22)
b_22 = tf.Variable(tf.truncated_normal([512], mean=0, stddev=1.0))
variable_summaries("b_22", b_22)
y_22 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_22, W_22), b_22))
x_2 = y_22
W_2 = tf.Variable(tf.truncated_normal([512, 256], mean=0, stddev=1.0))
variable_summaries("W_2", W_2)
b_2 = tf.Variable(tf.truncated_normal([256], mean=0, stddev=1.0))
variable_summaries("b_2", b_2)
y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_2, W_2), b_2))
x_1 = y_2
W_1 = tf.Variable(tf.truncated_normal([256, n_classes], mean=0, stddev=1.0))
variable_summaries("W_1", W_1)
b_1 = tf.Variable(tf.truncated_normal([n_classes], mean=0, stddev=1.0))
variable_summaries("b_1", b_1)
y_1_eval = tf.nn.bias_add(tf.matmul(x_1, W_1) , b_1)
y_1 = tf.nn.dropout(y_1_eval, dropout_keep_prob)
y_true = tf.placeholder(tf.float32, [None, n_classes])

f = lambda alpha: sess.run(alpha, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})

''' DEFINE LOSS FUNCTION '''
cross_entropy_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
tf.summary.scalar("cross_entropy", cross_entropy_2)
loss = cross_entropy_2 
tf.summary.scalar("loss", loss)


''' DEFINE OPTIMIZATION TECHNIQUE '''
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_1_eval, 1), tf.argmax(y_true, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)


''' TRAIN '''
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess = tf.Session()
train_writer = tf.summary.FileWriter('tensorboard_log6/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log6/test')
sess.run(init)
for i in range(60000 * 10):
    if i % 10 == 0:
        summary, acc = sess.run([merged,accuracy], feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
        test_writer.add_summary(summary, i)
        print(acc)
    else:
        batch_xs, batch_ys = cifar.train_next_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_true:batch_ys})
        train_writer.add_summary(summary, i)
    if i % 10000 == 0:
        print("LEARNING RATE DECAYED")
        learning_rate *= learning_rate_decay
