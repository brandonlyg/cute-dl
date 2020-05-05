# coding=utf-8

import sys
sys.path.append("../..")
sys.path.append("../../cutedl")

'''
加载MNIST数据集
'''

import pdb
import os
import struct

import numpy as np
from cutedl.dataset import Dataset

class Mnist(object):

    '''
    dir 数据集所在目录, 默认在./mnist目录下
    '''
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.dirname(__file__) + "/mnist"
            if dir == '/mnist':
                dir = '.'+dir

        self.__dir = dir

    '''
    加载数据集
    batch_size
    '''
    def load(self, batch_size, flatting=False, normalize=True):
        dir = self.__dir

        #读取训练数据集
        print("read train dataset")
        data = self.__read_image(dir+"/train-images-idx3-ubyte")
        label = self.__read_label(dir+"/train-labels-idx1-ubyte")
        train = self.__build_dataset(data, label, batch_size, flatting, normalize)
        #pdb.set_trace()
        #读取测试数据集
        print("read test dataset")
        data = self.__read_image(dir+"/t10k-images-idx3-ubyte")
        label = self.__read_label(dir+"/t10k-labels-idx1-ubyte")
        test = self.__build_dataset(data, label, batch_size, flatting, normalize)

        print("return datasets")
        return train, test

    def __build_dataset(self, data, label, batch_size, flatting, normalize):
        #展平成2维
        if flatting:
            m = data.shape[0]
            data = data.reshape((m, -1))
            label = label.reshape((m, -1))
        #标准化
        if normalize:
            data = data/255

        ds = Dataset(data, label, batch_size)
        return ds

    '''
    读取图片数据
    '''
    def __read_image(self, fpath):
        with open(fpath, 'rb') as f:
            '''
            读取16bytes, 4个32bit整数, 依次为:
            magic number
            number of images
            number of columns
            '''
            d = f.read(16)
            _, m, h, w = struct.unpack("!4I", d)
            '''
            后面每28*28bytes为一张图片信息
            '''

            '''
            读取所有数据
            '''
            size = m * h * w
            d = f.read(size)
            arr = struct.unpack('%dB'%len(d), d)

            res = np.array(arr).reshape(m, 1, h, w)
            return res

    '''
    读取标签数据
    '''
    def __read_label(self, fpath):
        with open(fpath, 'rb') as f:
            '''
            读取8个字节, 2个32位整数, 依次为
            magic number
            number of items
            '''
            d = f.read(8)
            _, m = struct.unpack("!2I", d)

            #读取后面的m个标签
            d = f.read(m)
            arr = struct.unpack("%dB"%m, d)

            #one-hot编码
            #pdb.set_trace()
            res = np.zeros((m, 10))
            res[range(m), list(arr)] = 1

            return res


if __name__ == '__main__':
    mnist = Mnist()
    train, test = mnist.load(64, flatting=True)
    pdb.set_trace()
