import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
cifar= input_data.read_data_sets("MNIST_data/", one_hot=True)

''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 0.1
learning_rate_decay = 0.1
num_epochs_per_decay = 350
num_iter = 10000
batch_size = 100 
reg_coeff = 0.0005
epsilon = tf.Variable(0.000000000000001 * np.ones([n_classes]), dtype=tf.float32)
dropout_keep_prob = 0.75

''' HELPER FUNCTION '''

''' DEFINE VARIABLES '''
x = tf.placeholder(tf.float32, [None, 784])
x_reshaped = tf.reshape(x, [-1, 28, 28, 1]) 

W_4 = tf.Variable(tf.random_uniform([5,5,1,64], 0, 100))
b_4 = tf.Variable(tf.random_uniform([64], 0, 100))
conv_4 = tf.nn.conv2d(x_reshaped, W_4, strides=[1,1,1,1], padding="VALID")
relu_4 = tf.nn.relu(tf.nn.bias_add(conv_4, b_4))
pooled_4 = tf.nn.max_pool(relu_4, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")
y_4 = tf.nn.local_response_normalization(pooled_4)
'''
filter_shape = [5,5,64,64]
W_3 = tf.Variable(tf.random_uniform(filter_shape))
b_3 = tf.Variable(tf.random_uniform([64]))
conv_3 = tf.nn.conv2d(y_4, W_3, strides=[1,1,1,1], padding="VALID")
conv_3 = tf.nn.local_response_normalization(conv_3)
relu_3 = tf.nn.relu(tf.nn.bias_add(conv_3, b_3))
pooled_3 = tf.nn.max_pool(relu_3, ksize=[1,3,3,1], strides=[1,1,1,1], padding="VALID")
y_3 = tf.reshape(pooled_3, [-1, 16*16*64])
'''
x_22 = tf.reshape(y_4, [-1, 22*22*64])
W_22 = tf.Variable(tf.random_uniform([22*22*64,512], 0, 100, dtype=tf.float32, seed=0))
b_22 = tf.Variable(tf.random_uniform([512], 0, 1, dtype=tf.float32, seed=0))
y_22 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_22, W_22), b_22))
x_2 = y_22
W_2 = tf.Variable(tf.random_uniform([512, 256], 0, 100, dtype=tf.float32, seed=0))
b_2 = tf.Variable(tf.random_uniform([256], 0, 1, dtype=tf.float32, seed=0))
y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_2, W_2), b_2))
x_1 = y_2
W_1 = tf.Variable(tf.random_uniform([256, n_classes], 0, 1, dtype=tf.float32, seed=0))
b_1 = tf.Variable(tf.random_uniform([n_classes], 0, 1, dtype=tf.float32, seed=0))
y_1_eval = tf.nn.bias_add(tf.matmul(x_1, W_1) , b_1)
y_1 = tf.nn.dropout(y_1_eval, dropout_keep_prob)
y_true = tf.placeholder(tf.float32, [None, n_classes])

f = lambda alpha: sess.run(alpha, feed_dict={x: cifar.test.images, y_true: cifar.test.labels})
''' DEFINE LOSS FUNCTION '''
# try use this loss function tf.nn.log_poisson_loss
cross_entropy_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
regularization = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) + tf.nn.l2_loss(W_22) + tf.nn.l2_loss(b_22) + tf.nn.l2_loss(W_4) + tf.nn.l2_loss(b_4)
loss = cross_entropy_2 + regularization * reg_coeff


''' DEFINE OPTIMIZATION TECHNIQUE '''
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_1_eval, 1), tf.argmax(y_true, 1)) #TODO: IS THIS THE RIGHT TESTING
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
''' TRAIN '''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(num_iter):
    batch_xs, batch_ys = cifar.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_true:batch_ys})
    if i%100== 0:
        print(f(loss))
        print(sess.run(accuracy, feed_dict={x: batch_xs, y_true: batch_ys}))
        print(sess.run(accuracy, feed_dict={x: cifar.test.images[:100], y_true: cifar.test.labels[:100]}))
    if i%1000 == 0:
        learning_rate *= 0.5

''' TEST '''
correct_prediction = tf.equal(tf.argmax(y_1_eval, 1), tf.argmax(y_true, 1)) #TODO: IS THIS THE RIGHT TESTING
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: cifar.test.images, y_true: cifar.test.labels}))


''' DEBUG 
y1= sess.run(y_1_eval, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
x1 = sess.run(x_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
w1 = sess.run(W_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
b1 = sess.run(b_1, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
y2 = sess.run(y_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
x2 = sess.run(x_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
w2 = sess.run(W_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
b2 = sess.run(b_2, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
'''
