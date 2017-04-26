'''input file is a dictionary with
1) 10000x3072 numpy array of uint8
    each row is a 32x32 image 
    first 1024 of the row is red channel values
    next is green
    last is blue 
2) 10000 list of numbers from 0-9 
    labels of the image 
this is code for python2    

stride - how much you move right or down after conv
padding - pads the sides with 0's. keeps the size of the image the same 
'''
import random
import numpy as np
import time
import tensorflow as tf
folder = "CIFAR_data/"

device_name = "/gpu:0"



class Cifar:

# shuffle data and pick 100
# randomly pick 100
    def __init__(self):
        with tf.device(device_name):

            s1 = unpickle(folder+"data_batch_1")
            s2 = unpickle(folder+"data_batch_2")
            s3 = unpickle(folder+"data_batch_3")
            s4 = unpickle(folder+"data_batch_4")
            s5 = unpickle(folder+"data_batch_5")
            test_image(s1)
            self.s5 = s5
            start = time.time()
            s1 = zip(apply_RGB_subtraction(s1['data']), s1['labels'])
            s2 = zip(apply_RGB_subtraction(s2['data']), s2['labels'])
            s3 = zip(apply_RGB_subtraction(s3['data']), s3['labels'])
            s4 = zip(apply_RGB_subtraction(s4['data']), s4['labels'])
            #s5 = zip(s5['data'], s5['labels'])
            s1.extend(s2)
            s1.extend(s3)
            s1.extend(s4)
            #s1.extend(s5)
            random.shuffle(s1)
            elapsedTime = time.time() - start # t = 0.04
            self.train_data = s1

            self.test_images = apply_RGB_subtraction(s5['data'])
            self.test_labels = one_hot(s5['labels'])
            #st = unpickle(folder+"test_batch")
            #self.test_images = st['data']/255.0
            #self.test_labels = one_hot(st['labels'])
     
    def train_next_batch(self, batch_size):
        with tf.device(device_name):
            start = time.time()
            train_data = self.train_data
            n = len(train_data)
            data = []; labels = []; picked = set()
            for _ in range(batch_size):
                index = random.randint(0, n - 1)
                while (index in picked):
                    index = random.randint(0, n - 1)
                data.append(train_data[index][0])
                labels.append(train_data[index][1])
                picked.add(index)
            elapsedTime = time.time() - start # t = 0.0004
            return np.array(data), one_hot(labels) 
            #return self.s[index]['data']/255.0, one_hot(self.s[index]['labels'])

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


def apply_RGB_subtraction(arr):
    # arr has 3072 values, first 1024 are red, next green, then blue 
    arr_T = arr.T
    n = 1024
    for i in range(3 * n):
        if i < n:
            arr_T[i] -= 122
        elif i < 2 * n:
            arr_T[i] -= 116
        else:
            arr_T[i] -= 104

    # for j in range(10000):
    #     for i in range(1024):
    #         arr[j][i] -= 122
    #     for i in range(1024, 2048):
    #         arr[j][i] -= 116
    #     for i in range(2048, 3072):
    #         arr[j][i] -= 104
    return arr

def one_hot(lst): 
    # lst: a list of labels
    ret = []
    for index in lst:
        sub_array = np.zeros([10])
        sub_array[index] += 1
        ret.append(sub_array)
    return np.array(ret)
    
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def test_image(s):
    image = s['data'][4:100]
    image_reshaped = np.reshape(image, [-1, 3, 32,32])
    image_reshaped = np.swapaxes(image_reshaped, 1,3)
    image_reshaped = np.swapaxes(image_reshaped, 2,1)
    tf.summary.image("No Preprocessing", image_reshaped)
