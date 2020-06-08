# coding=utf-8

import pdb
import numpy as np

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

        self.__indices = None

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

    #打乱
    def shuffle(self):
        if self.__indices is None:
            self.__indices = np.arange(self.__data.shape[0])

        np.random.shuffle(self.__indices)

    def as_iterator(self):
        bs = self.__batch_size
        bc = self.__batch_count
        data = self.__data
        label = self.__label

        if self.__indices is not None:
            data = self.__data[self.__indices]
            label = self.__label[self.__indices]

        for i in range(bc):
            start = i * bs
            end = start + bs
            yield (data[start:end],label[start:end])


'''
由参差不齐的张量构成的数据集
'''
class RaggedDataset:
    '''
    data 数据list
    labels 标签np.array
    batches 数据集的最大批次数 <=0 表示使用所有数据
    '''
    def __init__(self, datas, labels, batch_size, batches=0, padding=0):
        if labels is not None and len(datas) != len(labels):
            raise Exception("invalid datas and labels count not equal.")

        '''
        对数据进行分批和填充
        '''
        #打乱
        m = len(datas)
        indices = np.arange(m)
        np.random.shuffle(indices)

        batch_datas = []
        batch_labels = []
        #分批, 填充
        bs = 0
        while True:
            #得到一批的索引
            start = bs * batch_size
            end = start + batch_size
            b_ids = indices[start:end]
            bs += 1

            #得到一批数据
            b_data = []
            b_label = None
            max_len = 0
            for idx in b_ids:
                item = datas[idx]
                b_data.append(item)
                if len(item) > max_len:
                    max_len = len(item)

            b_label = labels[b_ids]
            #pdb.set_trace()
            #填充
            pads = [padding] * max_len
            for i in range(len(b_data)):
                d = b_data[i]
                if len(d) == max_len:
                    continue

                d = d + pads[:(max_len-len(d))]
                b_data[i] = d

            batch_datas.append(np.array(b_data))
            batch_labels.append(b_label)

            if b_ids.shape[0] != batch_size:
                #不足一批
                break

        self.__datas = batch_datas
        self.__labels = batch_labels

        if batches <= 0:
            batches = len(datas)

        self.__datas = self.__datas[0:batches]
        self.__labels = self.__labels[0:batches]

        self.__indices = None
        self.shuffle()

    @property
    def batch_count(self):
        return len(self.__datas)

    @property
    def data(self):
        return self.__datas

    @property
    def label(self):
        return self.__labels

    #打乱
    def shuffle(self):
        if self.__indices is None:
            self.__indices = np.arange(len(self.__datas))

        np.random.shuffle(self.__indices)

    def as_iterator(self):
        for i in self.__indices:
            yield (self.__datas[i], self.__labels[i])
