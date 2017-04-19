import tensorflow as tf
from data_generator import Cifar

c = Cifar()

x = tf.placeholder(tf.float32, [None, 3072])
x = tf.reshape(x, [100, 32, 32, 3])
W = tf.Variable(tf.constant(0.1, shape=[5,5,3,64]))
conv = tf.nn.conv2d(x, W, [1,1,1,1], padding="SAME")




init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
p = sess.run(conv, feed_dict={x:c.train_next_batch(100)[0]})
print(p)
