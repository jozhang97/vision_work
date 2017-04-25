import tensorflow as tf
import random
import numpy as np
import helper_variable_generation as hvg
from data_generator import Cifar

device_name = "/gpu:0"

with tf.device(device_name):
    cifar = Cifar()

    ''' HYPERPARAMETERS '''
    n_classes = 10
    learning_rate = 1e-3
    learning_rate_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('learning_rate', learning_rate_placeholder)
    learning_rate_decay = 0.1
    WEIGHT_DECAY_FACTOR = 0.001

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
    DELAYS_PER_EPOCH = 1
    num_epochs_per_decay = 350

    batch_size = 100 
    reg_coeff = 0.00005
    dropout_keep_prob = tf.placeholder(tf.float32)

    ''' HELPER FUNCTIONS '''
    def add_to_collection_weights(W):   
        for val in W.values():
            tf.add_to_collection("weights", val)

    ''' DEFINE VARIABLES '''
    W = {
        "W_00": hvg.weight_variable([1000, n_classes]),
        "W_01": hvg.weight_variable([1,1]), 

        "W_02": hvg.weight_variable([3,3,512,512]),
        "W_03": hvg.weight_variable([3,3,512,512]),
        "W_04": hvg.weight_variable([3,3,512,512]), 
        "W_05": hvg.weight_variable([3,3,256,512]),

        "W_06": hvg.weight_variable([3,3,256,256]),
        "W_07": hvg.weight_variable([3,3,256,256]),
        "W_08": hvg.weight_variable([3,3,256,256]),
        "W_09": hvg.weight_variable([3,3,128,256]),

        "W_10": hvg.weight_variable([3,3,128,128]),
        "W_11": hvg.weight_variable([3,3,128,128]),
        "W_12": hvg.weight_variable([3,3,128,128]),
        "W_13": hvg.weight_variable([3,3,64,128]),

        "W_14": hvg.weight_variable([3,3,64,64]),
        "W_15": hvg.weight_variable([3,3,64,64]),
        "W_16": hvg.weight_variable([3,3,64,64]),
        "W_17": hvg.weight_variable([3,3,64,64]),

        "W_18": hvg.weight_variable([7,7,3,64]),
        }
    add_to_collection_weights(W)
    b = {
        "b_00": hvg.bias_variable([n_classes]),
        "b_01": hvg.bias_variable([1000]),

        "b_02": hvg.bias_variable([512]),
        "b_03": hvg.bias_variable([512]),
        "b_04": hvg.bias_variable([512]),
        "b_05": hvg.bias_variable([512]),

        "b_06": hvg.bias_variable([256]),
        "b_07": hvg.bias_variable([256]),
        "b_08": hvg.bias_variable([256]),
        "b_09": hvg.bias_variable([256]),

        "b_10": hvg.bias_variable([128]),
        "b_11": hvg.bias_variable([128]),
        "b_12": hvg.bias_variable([128]),
        "b_13": hvg.bias_variable([128]),

        "b_14": hvg.bias_variable([64]),
        "b_15": hvg.bias_variable([64]),
        "b_16": hvg.bias_variable([64]),
        "b_17": hvg.bias_variable([64]),

        "b_18": hvg.bias_variable([64]),
        }
    hvg.variable_summaries_map(W)
    hvg.variable_summaries_map(b)

    x_reshaped = tf.placeholder(tf.float32, [None, 32, 32, 3])
    tf.summary.image("image", x_reshaped)

    y_18 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(x_reshaped, W['W_18']), b['b_18']))
    print(y_18.get_shape())
    y_18 = hvg.max_pool_3x3(y_18)
    print(y_18.get_shape())

    y_17 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_18, W['W_17']), b['b_17']))
    print(y_17.get_shape())
    y_16 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_17, W['W_16']), b['b_16']) )
    print(y_16.get_shape())
    y_15 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_16, W['W_15']), b['b_15']))
    print(y_15.get_shape())
    y_14 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_15, W['W_14']), b['b_14']) )
    print(y_14.get_shape())

    y_13 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_14, W['W_13']), b['b_13']))
    print(y_13.get_shape())
    y_12 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_13, W['W_12']), b['b_12']))
    print(y_12.get_shape())
    y_11 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_12, W['W_11']), b['b_11']))
    print(y_11.get_shape())
    y_10 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_11, W['W_10']), b['b_10'])  )
    print(y_10.get_shape())

    y_09 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_10, W['W_09']), b['b_09']))
    y_08 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_09, W['W_08']), b['b_08'])  )
    y_07 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_08, W['W_07']), b['b_07']))
    y_06 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_07, W['W_06']), b['b_06'])  )

    y_05 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_06, W['W_05']), b['b_05']))
    y_04 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_05, W['W_04']), b['b_04'])  )
    y_03 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_04, W['W_03']), b['b_03']))
    y_02 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(y_03, W['W_02']), b['b_02'])  )

    y_02_pooled = hvg.avg_pool(y_02)

    dim = y_02_pooled.get_shape()[1].value
    W["W_01"] = hvg.weight_variable([dim * dim*512, 1000])
    y_02_reshaped = tf.reshape(y_02_pooled,[-1, dim*dim*512])
    print(dim)

    y_01 = tf.nn.bias_add(tf.matmul(y_02_reshaped, W["W_01"]) , b["b_01"])
    print(y_01.get_shape())
    y_01 = tf.nn.dropout(y_01, dropout_keep_prob)
    y_00 = tf.nn.bias_add(tf.matmul(y_01, W["W_00"]) , b["b_00"])

    y_true = tf.placeholder(tf.float32, [None, n_classes])



    ''' DEFINE LOSS FUNCTION '''

    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
          input_tensor = WEIGHT_DECAY_FACTOR*tf.stack(
              [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
          ),
          name='weights_norm'
      )
    tf.add_to_collection('losses', weights_norm)
    tf.summary.scalar('weights_norm', weights_norm)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_00, labels=y_true))
    tf.summary.scalar("cross_entropy", cross_entropy)
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss') 
    tf.summary.scalar("loss", loss)

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])

    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_00, 1), tf.argmax(y_true, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    ''' TRAIN '''
    merged = tf.summary.merge_all()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options, log_device_placement=False)
sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet/test')
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

