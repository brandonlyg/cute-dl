# coding=utf-8

'''
数据集类
'''

class Dataset(object):

    '''
    data 数据
    label 数据对应的标签
    '''
    def __init__(self, data, label, batch_size):
        self.__data = data
        self.__label = label
        self.__batch_size = batch_size

        #检查数据集, 第0维必须相等
        if self.__data.shape[0] != self.__label.shape[0]:
            raise Exception("invalid data set. dim 0 not equal")

        #得到可划分的批数
        self.__batch_count = data.shape[0]//self.__batch_size
        if data.shape[0] % self.__batch_size != 0:
            self.__batch_count += 1

    @property
    def batch_count(self):
        return self.__batch_count

    @property
    def data(self):
        return self.__data

    @property
    def label(self):
        return self.__label

    def as_iterator(self):
        bs = self.__batch_size
        bc = self.__batch_count
        for i in range(bc):
            yield (self.__data[i*bs:i*bs+bs], self.__label[i*bs:i*bs+bs])
