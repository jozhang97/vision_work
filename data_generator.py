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
folder = "CIFAR_data/"
class Cifar:
    def __init__(self):
        s1 = unpickle(folder+"data_batch_1")
        s2 = unpickle(folder+"data_batch_2")
        s3 = unpickle(folder+"data_batch_3")
        s4 = unpickle(folder+"data_batch_4")
        s5 = unpickle(folder+"data_batch_5")
        self.s = [s1,s2,s3,s4,s5]
        st = unpickle(folder+"test_batch")
        self.test_images = st['data']/255.0
        self.test_labels = one_hot(st['labels'])
     
    def train_next_batch(self, batch_size):
        index = int(random.random() * 5)  
        # TODO: put in batch_size
        return self.s[index]['data']/255.0, one_hot(self.s[index]['labels'])
 
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
