# coding=utf-8

import pdb

import numpy as np
from model import Layer, LayerParam
import utils

'''
同样神经网络层
'''

'''
全连接层
'''
class Dense(Layer):
    tag='Dense'

    def __init__(self, *outshape, **kargs):
        #print("Dense kargs:", kargs)


        #参数
        self.__W = None
        self.__b = None

        super().__init__(*outshape, **kargs)

        #输入数据的形状
        self.__in_shape = None
        #输入数据, 已经转换成合适的形状
        self.__in_batch = None

    def init_params(self):

        #展平纬度, 初始化参数值
        std = 0.01
        shape = (utils.flat_shape(self.inshape), utils.flat_shape(self.outshape))
        wval = np.random.randn(shape[0], shape[1]) * std
        bval = np.zeros(shape[1])

        self.__W = LayerParam(self.name, 'W', wval)
        self.__b = LayerParam(self.name, 'b', bval)

    @property
    def params(self):
        return [self.__W, self.__b]

    def forward(self, in_batch, training=False):
        W = self.__W.value
        b = self.__b.value

        wshape = W.shape
        m = in_batch.shape[0]
        self.__in_shape = in_batch.shape

        #如果输入输入形状不和适, 转换成期望的形状
        in_x = in_batch
        if self.__in_shape != (m, wshape[0]):
            in_x = in_x.reshape((-1, wshape[0]))

        self.__in_batch = in_x

        out = in_x @ W + b

        return self.activation(out)

    '''
    反向传播梯度
    gradient shape=(m,outshape)
    '''
    def backward(self, gradient):
        W = self.__W.value

        #print("gradient shape:", gradient.shape)
        grad = self.activation.grad(gradient)

        #参数梯度
        #(inshape, outshape) = (inshape, m) @ (m, outshape)
        self.__W.gradient = self.__in_batch.T @ grad

        self.__b.gradient = grad.sum(axis=0)
        #pdb.set_trace()
        #输入数据梯度 (m,inshape) = (m,outshape) @ (outshape, inshape)
        grad_in_x = grad @ W.T

        #如果有需要的话还原形状
        if grad_in_x.shape != self.__in_shape:
            #pdb.set_trace()
            grad_in_x = grad_in_x.reshape(self.__in_shape)

        return grad_in_x


    #重置当前层的状态
    def reset(self):
        self.__W.reset()
        self.__b.reset()

        self.__in_shape = None
        self.__in_batch = None


'''
dropout层
'''
class Dropout(Layer):
    tag='Dropout'

    '''
    drop: 随机丢弃比例
    '''
    def __init__(self, *outshape, **kargs):
        self.__drop = 0.0

        if 'drop' in kargs:
            self.__drop == kargs['drop']

        super().__init__(-1, **kargs)

    def init_params(self):
        pass

    @property
    def params(self):
        return []

    def forward(self, in_batch, training=False):
        if not training or self.__drop < 1e-8:
            return in_batch

        #生成[0, 1)直接的均价分布
        tmp = np.random.uniform(size=in_batch.shape).reshape(-1)
        #得到丢弃的索引并随机打乱
        indices = tmp <= self.__drop
        indices = indices.reshape(in_batch.shape)
        #丢弃数据
        in_batch[indices] = 0

        return in_batch

    def backward(self, gradient):
        return gradient
