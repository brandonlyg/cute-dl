# coding=utf-8

import sys
sys.path.append("../..")
sys.path.append("../../cutedl")

'''
cifar-10数据集
'''
import pdb
import glob
import pickle
import os
import numpy as np
from cutedl.dataset import Dataset

class Cifar10(object):

    '''
    dir 数据集所在目录, 默认在./cifar-10目录下
    '''
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.dirname(__file__) + "/cifar-10"
            if dir == '/cifar-10':
                dir = '.'+dir

        self.__test_file = dir+"/test_batch"
        self.__train_files = glob.glob(dir+"/data_batch_*")

    '''
    加载数据集
    batch_size
    '''
    def load(self, batch_size, normalize=True):
        #加载训练数据集
        data = None
        label = None
        print("load cifar10 train dataset")
        for fpath in self.__train_files:
            d, l = self.__read_file(fpath)
            #pdb.set_trace()
            if data is None:
                data = d
                label = l
            else:
                data = np.vstack((data, d))
                label = np.hstack((label, l))

        train = self.__build_dataset(data, label, batch_size, normalize)

        #加载测试数据集
        print("load cifar10 test dataset")
        data, label = self.__read_file(self.__test_file)
        test = self.__build_dataset(data, label, batch_size, normalize)

        print("load cifar finished")
        return train, test

    def __read_file(self, fpath):
        fdata = None
        with open(fpath, 'rb') as f:
            fdata = pickle.load(f, encoding='bytes')

        '''
        fdata.keys()
        dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        
        '''
        data = fdata[b'data']
        label = fdata[b'labels']

        return data, label

    def __build_dataset(self, data, label, batch_size, normalize):
        #pdb.set_trace()
        if normalize:
            data = data/255

        m, _ = data.shape
        data = data.reshape((m, 3, 32, 32))

        #label one-hot编码
        onehot = np.zeros((m, 10))
        onehot[range(m), label] = 1

        ds = Dataset(data, onehot, batch_size)
        return ds


import matplotlib.pyplot as plt

if '__main__' == __name__:
    c10 = Cifar10()
    train, test = c10.load(64, normalize=False)

    train.shuffle()
    batch, label = next(train.as_iterator())
    #pdb.set_trace()
    train_img = np.moveaxis(batch[0], 0, -1)

    test.shuffle()
    batch, label = next(test.as_iterator())
    test_img  = np.moveaxis(batch[0], 0, -1)

    plt.subplot(2,1,1)
    plt.imshow(train_img)

    plt.subplot(2,1,2)
    plt.imshow(test_img)

    plt.savefig("cifar-img.png")
