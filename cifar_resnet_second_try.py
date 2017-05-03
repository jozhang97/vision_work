import tensorflow as tf
import random
import numpy as np
import helper_variable_generation as hvg
from data_generator import Cifar

device_name = "/gpu:0"
# needed 30k iterations to start seeing results
# 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1 really fast

with tf.device(device_name):
    cifar = Cifar()

    n = 4
    ''' HYPERPARAMETERS '''
    n_classes = 10
    learning_rate = 1e-3
    learning_rate_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('learning_rate', learning_rate_placeholder)
    learning_rate_decay = 0.1
    WEIGHT_DECAY_FACTOR = 1e-5
    WEIGHT_DECAY_FACTOR_placeholder = tf.placeholder(tf.float32)
    MOMENTUM = 0.9
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
    DELAYS_PER_EPOCH = 1
    RESTORE_WEIGHTS = False
    batch_size = 128 
    dropout_keep = 0.5
    dropout_keep_prob = tf.placeholder(tf.float32)

    ''' HELPER FUNCTIONS '''  
    def add_to_collection_weights(W):   
        for val in W.values():
            tf.add_to_collection("weights", val)

    def convert_images_into_2D(images):
        images = tf.reshape(images, [-1, 3, 32, 32])
        images = tf.transpose(images, [0, 2, 3, 1])
        return images 

    def distort_images(images, num_images):
        new_images = []
        for i in range(16):
            new_images.append(distort(images[i]))
        return np.array(new_images)

    def crop(image, height = 28, weight = 28):
        resized_image = tf.random_crop(reshaped_image, [height, width, 3])
        #ret_image = tf.image.per_image_whitening(resized_image)
        ret_image = resized_image
        return ret_image

    def distort(reshaped_image, height = 28, weight = 28):
      # Randomly crop a [height, width] section of the image.
      distorted_image = reshaped_image
      # distorted_image = tf.random_crop(_image, [height, width, 3])
      if random.random() > 0.5:
          return reshaped_image

      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(distorted_image)

      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
      #ret_image = tf.image.per_image_whitening(distorted_image)
      ret_image = distorted_image
      return ret_image 

    def add_residual(small, big):
        # make sure same number of channels
        small_shape = small.get_shape().as_list()
        big_shape = big.get_shape().as_list()
        # assert small_shape[3] == big_shape[3]
        x_diff = big_shape[1] - small_shape[1]
        y_diff = big_shape[2] - small_shape[2]
        chan_diff = -1 * (big_shape[3] - small_shape[3])
        if x_diff != 0 or y_diff != 0:
            small = tf.pad(small, [[0, 0], [x_diff // 2, x_diff // 2 + x_diff%2], [y_diff // 2, y_diff // 2 + y_diff %2], [0,0]], "CONSTANT")
            big = tf.pad(big, [[0, 0], [0, 0], [0, 0], [chan_diff//2 , chan_diff//2 + chan_diff%2]])
            return small + big
        if chan_diff != 0:
            big = tf.pad(big, [[0, 0], [0, 0], [0, 0], [chan_diff//2 , chan_diff//2 + chan_diff%2]])
        small_normed = tf.nn.local_response_normalization(small)
        big_normed = tf.nn.local_response_normalization(big)
        return small_normed + big_normed

    def restore_weights():
        new_saver = tf.train.import_meta_graph('trained/current_resnet_model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('trained/'))
        all_vars = tf.get_collection('weights')
        for v in all_vars:
            v_ = sess.run(v)

    def residual(input, inc_dim = False):
        # get shape of input
        shape = input.get_shape().as_list()
        in_channels = shape[3]
        
        # determine number of channels and stride
        if inc_dim:
            stride1=2
            out_channels = 2 * in_channels
        else:
            stride1=1
            out_channels = in_channels
        stride2=2

        # initialize weights
        W_1 = hvg.weight_variable([3,3,in_channels,out_channels], stddev=0.2)
        b_1 = hvg.bias_variable([out_channels], constant=0.1)
        W_2 = hvg.weight_variable([3,3,out_channels,out_channels], stddev= 0.2)
        b_2 = hvg.bias_variable([out_channels], constant=0.1)

        # first conv layer
        conv1 = hvg.conv2d(input, W_1, stride=stride1)
        bias1 = tf.nn.bias_add(conv1, b_1)
        relu1 = tf.nn.relu(bias1)

        # second conv layer
        conv2 = hvg.conv2d(relu1, W_2, stride=stride2)
        bias2 = tf.nn.bias_add(conv2, b_2)

        # apply average pool if incrasing dimensions
        if inc_dim:
            bias2 = hvg.avg_pool_3x3(bias2, stride=2)

        # add residual
        ret = add_residual(bias2, input)
        return ret



    num_images = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    x_reshaped = convert_images_into_2D(x)
    x_reshaped = tf.pad(x_reshaped, [[0,0], [3,3], [3,3], [0,0]], "CONSTANT")
    tf.summary.image("image", x_reshaped)

    # first conv layer
    W_first = hvg.weight_variable([7,7,3,64], stddev=1.0)
    b_first = hvg.bias_variable([64])
    conv1 = tf.nn.relu(tf.nn.bias_add(hvg.conv2d(x_reshaped, W_first), b_first))

    # max pool layer
    res = hvg.max_pool_3x3(conv1)

    # first residual layers
    for i in range(n):
        res = residual(res)

    # first increasing residual layer
    res = residual(res, inc_dim=True)

    # second residual layer
    for i in range(n):
        res = residual(res)

    # second increasing residual layer
    res = residual(res, inc_dim=True)

    # third residual layer
    for i in range(n):
        res = residual(res)  

    # apply relu
    relu = tf.nn.relu(res)

    # apply average pool
    avg_pool1 = hvg.avg_pool_3x3(relu)
    #avg_pool1 = tf.nn.dropout(avg_pool1, dropout_keep_prob)


    # calc size of feature vector and reshape
    dim = avg_pool1.get_shape()[1].value
    num_channels = avg_pool1.get_shape().as_list()[3]
    print("Size after convolution", dim, num_channels)
    reshaped = tf.reshape(avg_pool1,[-1, dim*dim*num_channels])
    
    # set up fc weights
    W_fc = hvg.weight_variable([dim * dim * num_channels, 10], stddev=1.0)
    b_fc = hvg.bias_variable([10], constant=0.1)
    tf.add_to_collection("weight_decay", W_fc)

    # apply fc layer
    y_01 = tf.nn.bias_add(tf.matmul(reshaped, W_fc) , b_fc)
    y_00 = y_01

    y_true = tf.placeholder(tf.float32, [None, n_classes])



    ''' DEFINE LOSS FUNCTION '''

    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
          input_tensor = WEIGHT_DECAY_FACTOR_placeholder*tf.stack(
              [tf.nn.l2_loss(i) for i in tf.get_collection('weight_decay')]
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
    ''' DEFINE OPTIMIZATION TECHNIQUE '''
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate_placeholder).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_00, 1), tf.argmax(y_true, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    ''' TRAIN '''
    merged = tf.summary.merge_all()
'''
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss])
'''


saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options, log_device_placement=False)
sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet_second_try/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_log_cifar_resnet_second_try/test')
sess.run(tf.global_variables_initializer())
if RESTORE_WEIGHTS:
    restore_weights()
for i in range(100000):
    if i % 100 == 0 and i > 100:
        offset = random.randint(0, 49)
        summary, acc = sess.run([merged,accuracy], feed_dict={x: cifar.test_images[offset*16:offset*16+16], y_true: cifar.test_labels[offset*16:offset*16+16], dropout_keep_prob: 1, learning_rate_placeholder: learning_rate, num_images: 16, WEIGHT_DECAY_FACTOR_placeholder: WEIGHT_DECAY_FACTOR})
        test_writer.add_summary(summary, i)
        print(acc)
    else:
        batch_xs, batch_ys = cifar.train_next_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_true:batch_ys, dropout_keep_prob: dropout_keep, learning_rate_placeholder: learning_rate, num_images: batch_size, WEIGHT_DECAY_FACTOR_placeholder: WEIGHT_DECAY_FACTOR})
        train_writer.add_summary(summary, i)
    if i == 15000 or i == 32000 or i == 48000:
        learning_rate *= learning_rate_decay
        print("Dropping learning rate")
    if i % 50000 == 0 and i > 0:
        saver.save(sess, 'trained/current_resnet_model')
    if i == 6000 or i == 18000:
        dropout_keep *= 0.5
        WEIGHT_DECAY_FACTOR *= 0.01





'''
fixed memory issue by making validation set smaller
post conv/pool, image is 8x8
'''
