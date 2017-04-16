import tensorflow as tf
import numpy as np
from data_generator import Cifar
cifar = Cifar()

# HYPERPARAMETERS
n_classes = 10
learning_rate = 0.1
num_iter = 100
batch_size = 100
reg_coeff = 0.5
epsilon = tf.Variable(0.000000000000001 * np.ones([n_classes]), dtype=tf.float32)

# DEFINE VARIABLES
x = tf.placeholder(tf.float32, [None, 3072])
y_true = tf.placeholder(tf.float32, [None, n_classes])
W = tf.Variable(tf.random_uniform([3072, n_classes], 0, 1, dtype=tf.float32, seed=0))
b = tf.Variable(tf.random_uniform([n_classes], 0, 1, dtype=tf.float32, seed=0))
pre = tf.matmul(x,W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# DEFINE LOSS FUNCTION
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + epsilon), reduction_indices=[1]))
mean_squared = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, y_true)))) 
regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

#loss = reg_coeff * regularization + cross_entropy 
loss = reg_coeff * regularization + mean_squared

# DEFINE OPTIMIZATION TECHNIQUE
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# TRAIN
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(num_iter):
    batch_xs, batch_ys = cifar.train_next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_true:batch_ys})

# TEST
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1)) #TODO: IS THIS THE RIGHT TESTING
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: cifar.test_images, y_true: cifar.test_labels}))


# DEBUG
myY= sess.run(y, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
myX = sess.run(x, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
myW = sess.run(W, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})
myB = sess.run(b, feed_dict={x: cifar.test_images, y_true: cifar.test_labels})


