import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
placeholder is where we put in data
variable is what we train
combine these to create output layer 
use output layer and placeholder to calculate loss

optimizer to minimize loss 
initialize variables and run session
run many iterations of the training alg on BATCHES OF DATA
'''

# SET UP VARIABLES
# None can be any dimensio aka any number of samples 
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

# DEFINE LOSS FUNCTION
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# DEFINE OPTIMIZATION TECHNIQUE
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

# TRAINING TIME
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    b = sess.run(b);
    print(b);

#VALIDATION TIME 

# TEST TIME
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
