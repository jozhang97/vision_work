import tensorflow as tf
import random
import numpy as np
import helper_variable_generation as hvg
from data_generator import Cifar
cifar = Cifar()

device_name = "/gpu:0"
''' HYPERPARAMETERS '''
n_classes = 10
learning_rate = 1e-3
learning_rate_placeholder = tf.placeholder(tf.float32)
tf.summary.scalar('learning_rate', learning_rate_placeholder)
learning_rate_decay = 0.1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
DELAYS_PER_EPOCH = 1
num_epochs_per_decay = 350

batch_size = 100 
reg_coeff = 0.00005
dropout_keep_prob = tf.placeholder(tf.float32)


with tf.device(device_name):
    ''' DEFINE VARIABLES '''
    W = {"W_1": hvg.weight_variable([192, n_classes]),
        "W_2": hvg.weight_variable([384, 192]), 
        "W_3": hvg.weight_variable([dim, 384]),
        "W_4": hvg.weight_variable([5, 5, 64, 64]), 
        "W_5": hvg.weight_variable([5,5,3,64]),
        }
    b = {"b_1": hvg.bias_variable([n_classes]),
        "b_2": hvg.bias_variable([192]),
        "b_3": hvg.bias_variable([384]),
        "b_4": hvg.bias_variable([64]),
        "b_5": hvg.bias_variable([64]),
        }
    hvg.variable_summaries_map(W)
    hvg.variable_summaries_map(b)

    x = tf.placeholder(tf.float32, [None, 3072])
    x_reshaped = tf.reshape(x, [-1, 32, 32, 3]) 
    tf.summary.image("image", x_reshaped)
    y_5 = tf.nn.local_response_normalization(yhvg.max_pool_3x3(tf.nn.relu(tf.nn.bias_add(hvg.conv2d(x_reshaped, W["W_5"]), b["b_5"]))))

    y_4 = hvg.max_pool_2x2(tf.nn.local_response_normalization(tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_5, W["W_4"]), b["b_4"]))))
    dim = y_4.get_shape()[1].value
    W["W_3"] = hvg.weight_variable([dim, 384])
    y_4 = tf.reshape(y_4, [-1, dim*dim*64])

    y_3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_4, W["W_3"]), b["b_3"]))

    y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_3, W["W_2"]), b["b_2"]))
    y_2 = tf.nn.dropout(y_2, dropout_keep_prob)
    y_1 = tf.nn.bias_add(tf.matmul(y_2, W["W_1"]) , b["b_1"])
    y_true = tf.placeholder(tf.float32, [None, n_classes])

    # W = {
    #     "W_1": hvg.weight_variable([256, n_classes]),
    #     "W_2": hvg.weight_variable([1024, 256]), 
    #     "W_3": hvg.weight_variable([8*8*64, 1024]), 
    #     "W_4": hvg.weight_variable([5, 5, 64, 64]), 
    #     "W_5": hvg.weight_variable([5, 5, 3, 64]), 
    #     }
    # b = {
    #     "b_1": hvg.bias_variable([n_classes]),
    #     "b_2": hvg.bias_variable([256]),
    #     "b_3": hvg.bias_variable([1024]),
    #     "b_4": hvg.bias_variable([64]),
    #     "b_5": hvg.bias_variable([64]),
    #     }
    # hvg.variable_summaries_map(W)
    # hvg.variable_summaries_map(b)

    # x = tf.placeholder(tf.float32, [None, 3072])
    # x_reshaped = tf.reshape(x, [-1, 32, 32, 3]) 
    # tf.summary.image("image", x_reshaped)
    # y_5 = tf.nn.local_response_normalization(hvg.max_pool_2x2(tf.nn.relu(tf.nn.bias_add(hvg.conv2d(x_reshaped, W["W_5"]), b["b_5"]))))

    # y_4 = hvg.max_pool_2x2(tf.nn.local_response_normalization(tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_5, W["W_4"]), b["b_4"]))))
    # y_4 = tf.reshape(y_4, [-1, 8*8*64])

    # y_3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_4, W["W_3"]), b["b_3"]))

    # y_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_3, W["W_2"]), b["b_2"]))
    # y_2 = tf.nn.dropout(y_2, dropout_keep_prob)
    # y_1 = tf.nn.bias_add(tf.matmul(y_2, W["W_1"]), b["b_1"])
    # y_true = tf.placeholder(tf.float32, [None, n_classes])


    ''' DEFINE LOSS FUNCTION '''
    cross_entropy_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_1, labels=y_true))
    tf.summary.scalar("cross_entropy", cross_entropy_2)
    loss = cross_entropy_2 
    tf.summary.scalar("loss", loss)


    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_1, 1), tf.argmax(y_true, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


''' TRAIN '''
merged = tf.summary.merge_all()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)
sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('tensorboard_log_cifar1/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_cifar1/test')
sess.run(tf.global_variables_initializer())
for i in range(60000 * 10):
    if i % 100 == 0:
        offset = random.randint(0, 499)
        summary, acc = sess.run([merged,accuracy], feed_dict={x: cifar.test_images[offset*16:offset*16+16], y_true: cifar.test_labels[offset*16:offset*16+16], dropout_keep_prob: 1, learning_rate_placeholder: learning_rate})
        test_writer.add_summary(summary, i)
        print(acc)
    else:
        batch_xs, batch_ys = cifar.train_next_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_true:batch_ys, dropout_keep_prob: 0.5, learning_rate_placeholder: learning_rate})
        train_writer.add_summary(summary, i)
    if i % 10000 == 0 and i > 0:
        learning_rate *= learning_rate_decay




'''
fixed memory issue by making validation set smaller
post conv/pool, image is 8x8
'''
