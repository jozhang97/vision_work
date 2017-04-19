import tensorflow as tf
import numpy as np
from data_generator import Cifar
cifar = Cifar()

''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 0.009
num_iter = 1000
batch_size = 100
reg_coeff = 0.05
epsilon = tf.Variable(0.000000000000001 * np.ones([n_classes]), dtype=tf.float32)
dropout_keep_prob = 0.5

''' HELPER FUNCTION '''

''' DEFINE VARIABLES '''
x = tf.placeholder(tf.float32, [None, 3072])
x_reshaped = tf.reshape(x, [-1, 32, 32, 3]) 

W_4 = tf.Variable(tf.truncated_normal([5,5,3,32], stddev=0.1))
b_4 = tf.Variable(tf.truncated_normal([32], stddev = 0.1))
conv_4 = tf.nn.conv2d(x_reshaped, W_4, strides=[1,1,1,1], padding="VALID")
relu_4 = tf.nn.relu(tf.nn.bias_add(conv_4, b_4))
pooled_4 = tf.nn.max_pool(relu_4, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")

filter_shape = [5,5,32,64]
W_3 = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_3 = tf.nn.conv2d(pooled_4, W_3, strides=[1,1,1,1], padding="VALID")
relu_3 = tf.nn.relu(tf.nn.bias_add(conv_3, b_3))
pooled_3 = tf.nn.max_pool(relu_3, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")
y_3 = tf.reshape(pooled_3, [-1, 20*20*64]) 




x_22 = y_3
W_22 = tf.Variable(tf.random_uniform([25600, 5120], 0, 1, dtype=tf.float32, seed=0))
b_22 = tf.Variable(tf.random_uniform([5120], 0, 1, dtype=tf.float32, seed=0))
y_22 = tf.nn.relu(tf.matmul(x_2, W_22) + b_22)
x_2 = y_22
W_2 = tf.Variable(tf.random_uniform([5120, 1024], 0, 1, dtype=tf.float32, seed=0))
b_2 = tf.Variable(tf.random_uniform([1024], 0, 1, dtype=tf.float32, seed=0))
y_2 = tf.nn.relu(tf.matmul(x_2, W_2) + b_2)
x_1 = y_2
W_1 = tf.Variable(tf.random_uniform([1024, n_classes], 0, 1, dtype=tf.float32, seed=0))
b_1 = tf.Variable(tf.random_uniform([n_classes], 0, 1, dtype=tf.float32, seed=0))
y_1 = tf.matmul(x_1, W_1) + b_1
y_1 = tf.nn.dropout(y_1, dropout)
y_true = tf.placeholder(tf.float32, [None, n_classes])

''' DEFINE LOSS FUNCTION '''
# try use this loss function tf.nn.log_poisson_loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_1 + epsilon), reduction_indices=[1]))
cross_entropy_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
mean_squared = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_1, y_true)))) 
regularization = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) + tf.nn.l2_loss(W_22) + tf.nn.l2_loss(b_22) + tf.nn.l2_loss(W_3) + tf.nn.l2_loss(b_3) + tf.nn.l2_loss(W_4) + tf.nn.l2_loss(b_4)
loss = reg_coeff * regularization + cross_entropy_2 

''' DEFINE OPTIMIZATION TECHNIQUE '''
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

''' TRAIN '''
dropout_keep_prob = 0.5
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(num_iter):
    batch_xs, batch_ys = cifar.train_next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_true:batch_ys})

''' TEST '''
dropout_keep_prob = 1
correct_prediction = tf.equal(tf.argmax(y_1, 1), tf.argmax(y_true, 1)) #TODO: IS THIS THE RIGHT TESTING
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: cifar.test_images, y_true: cifar.test_labels}))


''' DEBUG '''
y1= sess.run(y_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
x1 = sess.run(x_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
w1 = sess.run(W_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
b1 = sess.run(b_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
y2 = sess.run(y_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
x2 = sess.run(x_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
w2 = sess.run(W_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
b2 = sess.run(b_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
