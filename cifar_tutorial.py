import tensorflow as tf
import random
import numpy as np
import helper_variable_generation as hvg
from data_generator import Cifar

device_name = "/gpu:0"

# 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1 really fast

with tf.device(device_name):
    cifar = Cifar()

    ''' HYPERPARAMETERS '''
    n_classes = 10
    learning_rate = 1e-1
    learning_rate_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('learning_rate', learning_rate_placeholder)
    learning_rate_decay = 0.1
    WEIGHT_DECAY_FACTOR = 0.004
    MOMENTUM = 0.9

    batch_size = 128 
    dropout_keep_prob = tf.placeholder(tf.float32)

    ''' HELPER FUNCTIONS '''  
    def add_to_collection_weights(W):   
        for val in W.values():
            tf.add_to_collection("weights", val)

    def convert_images_into_2D(images):
        images = tf.reshape(images, [-1, 3, 32, 32])
        images = tf.transpose(images, [0, 2, 3, 1])
        return images 

    def distort_images(images):
        return images
        new_images = []
        for i in range(images.get_shape()[0]):
            distorted_image = distort(images[i])
            new_images.append(distorted_image)
            new_images.append(crop(image[i]))
        return np.array(new_images) 

    def crop(image, height = 24, weight = 24):
        resized_image = tf.random_crop(reshaped_image, [height, width, 3])
        ret_image = tf.image.per_image_whitening(resized_image)
        return ret_image

    def distort(reshaped_image, height = 24, weight = 24):
      # Randomly crop a [height, width] section of the image.
      distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(distorted_image)

      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
      ret_image = tf.image.per_image_whitening(distorted_image)
      return ret_image 

    def add_residual(small, big):
        # make sure same number of channels
        small_shape = small.get_shape().as_list()
        big_shape = big.get_shape().as_list()
        # assert small_shape[3] == big_shape[3]
        x_diff = big_shape[1] - small_shape[1]
        y_diff = big_shape[2] - small_shape[2]
        chan_diff = -1 * (big_shape[3] - small_shape[3])
        if chan_diff != 0:
            big = tf.pad(big, [[0, 0], [0, 0], [0, 0], [chan_diff//2 , chan_diff//2 + chan_diff%2]])
        #small = tf.pad(small, [[0, 0], [x_diff // 2, x_diff // 2 + x_diff%2], [y_diff // 2, y_diff // 2 + y_diff %2], [chan_diff // 2, chan_diff // 2]], "CONSTANT")
        small_normed = tf.nn.local_response_normalization(small)
        big_normed = tf.nn.local_response_normalization(big)
        return small_normed + big_normed

    ''' DEFINE VARIABLES '''
    W = {
        "W_sm": hvg.weight_variable([192, n_classes], stddev=1/192)
        "W_fc2": hvg.weight_variable([384, 192], stddev=0.04), #weight decay this by 0.004

        "W_conv2": hvg.weight_variable([5,5,64,64], stddev=5e-2),
        "W_conv1": hvg.weight_variable([5,5,3,64],stddev=5e-2),
        }
    add_to_collection_weights(W)
    b = {
        "b_sm": hvg.bias_variable([n_classes], 0),
        "b_fc2": hvg.bias_variable([192], 0.1),
        "b_fc1": hvg.bias_variable([384], 0.1),

        "b_conv2": hvg.bias_variable([64], 0.1),
        "b_conv1": hvg.bias_variable([64], 0),
        }
    hvg.variable_summaries_map(W)
    hvg.variable_summaries_map(b)

    # get the picture
    x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    x_reshaped = convert_images_into_2D(x)
    x_reshaped = distort_images(x_reshaped)
    tf.summary.image("image", x_reshaped)

    # apply first convolutional layer
    conv1 = hvg.conv2d(x_reshaped, W["W_conv1"], stride=1)
    bias1 = tf.nn.bias_add(conv1, b["b_conv1"])
    relu1 = tf.nn.relu(bias1)

    # apply first pool
    pool1 = hvg.max_pool_3x3(relu1, stride=2)

    # apply first normalization
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) 

    # apply second convolutional layer
    conv2 = hvy.conv2d(norm1, W["W_conv2"], stride=1)
    bias2 = tf.nn.bias_add(conv2, b["b_conv2"])
    relu2 = tf.nn.relu(bias2)

    # apply second normalization
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # apply second pool 
    pool2 = hvy.max_pool_3x3(norm2, stride=2)

    # apply first fc layer
    dim = pool2.get_shape()[1].value
    W["W_fc1"] = hvg.weight_variable([dim * dim*64, 384], std = 0.04) # this is decayed by wd = 0.004
    reshaped = tf.reshape(pool2,[-1, dim*dim*64])
    fc1 = tf.nn.bias_add(tf.matmul(reshaped, W["W_fc1"]) , b["b_fc1"])
    relu3 = tf.nn.relu(fc1)

    # apply second fc layer
    fc2 = tf.nn.bias_add(tf.matmul(relu3, W["W_fc2"]) , b["b_fc2"])
    relu4 = tf.nn.relu(fc2)

    # apply dropout
    dropout_ed = tf.nn.dropout(relu4, dropout_keep_prob)

    # apply softmax layer 
    sm = tf.nn.bias_add(tf.matmul(dropout_ed, W["W_sm"]) , b["b_sm"])
    
    # predicted label
    y_00 = sm

    # true label
    y_true = tf.placeholder(tf.float32, [None, n_classes])



    ''' DEFINE LOSS FUNCTION '''
    # Weight decay
    tf.add_to_collection("weight_decay", W["W_fc1"])
    tf.add_to_collection("weight_decay", W["W_fc2"])
    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
          input_tensor = WEIGHT_DECAY_FACTOR*tf.stack(
              [tf.nn.l2_loss(i) for i in tf.get_collection('weight_decay')]
          ),
          name='weights_norm'
      )
    tf.add_to_collection('losses', weights_norm)
    tf.summary.scalar('weights_norm', weights_norm)

    # Cross Entropy Loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_00, labels=y_true))
    tf.summary.scalar("cross_entropy", cross_entropy)
    tf.add_to_collection('losses', cross_entropy)

    # Total Loss
    loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss') 
    tf.summary.scalar("loss", loss)


    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate_placeholder).minimize(loss)

    ''' CALCULATE PREDICTION ACCURACY ''' 
    correct_prediction = tf.equal(tf.argmax(y_00, 1), tf.argmax(y_true, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    ''' TENSORBOARD PREP '''
    merged = tf.summary.merge_all()

'''
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])
'''


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options, log_device_placement=False)
sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet_tutorial/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet_tutorial/test')
sess.run(tf.global_variables_initializer())

''' TRAIN '''
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
    if i == 32000 or i == 48000:
        learning_rate *= learning_rate_decay




'''
fixed memory issue by making validation set smaller
post conv/pool, image is 8x8
'''
