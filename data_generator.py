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
import tensorflow as tf
import random
import numpy as np
import time
folder = "CIFAR_data/"
class Cifar:

# shuffle data and pick 100
# randomly pick 100
    def __init__(self):
        s1 = unpickle(folder+"data_batch_1")
        s2 = unpickle(folder+"data_batch_2")
        s3 = unpickle(folder+"data_batch_3")
        s4 = unpickle(folder+"data_batch_4")
        s5 = unpickle(folder+"data_batch_5")
        start = time.time()
        s1 = zip(s1['data'], s1['labels'])
        s2 = zip(s2['data'], s2['labels'])
        s3 = zip(s3['data'], s3['labels'])
        s4 = zip(s4['data'], s4['labels'])
        #s5 = zip(s5['data'], s5['labels'])
        s1.extend(s2)
        s1.extend(s3)
        s1.extend(s4)
        #s1.extend(s5)
        random.shuffle(s1)
        elapsedTime = time.time() - start # t = 0.04
        self.train_data = s1

        self.test_images = s5['data']/255.0
        self.test_labels = one_hot(s5['labels'])
        #st = unpickle(folder+"test_batch")
        #self.test_images = st['data']/255.0
        #self.test_labels = one_hot(st['labels'])
     
    def train_next_batch(self, batch_size):
        start = time.time()
        train_data = self.train_data
        n = len(train_data)
        data = []; labels = []; picked = []
        for _ in range(batch_size):
            index = random.randint(0, n - 1)
            while (index in picked):
                index = random.randint(0, n - 1)
            data.append(train_data[index][0])
            labels.append(train_data[index][1])
            picked.append(index)
        elapsedTime = time.time() - start # t = 0.0004
        return np.array(data)/255.0, one_hot(labels) 
        #return self.s[index]['data']/255.0, one_hot(self.s[index]['labels'])
 
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
