# coding=utf-8

import numpy as np
from cutedl.model import Layer, LayerParam
from cutedl import utils

'''
测试用例中使用的layer
'''

class Simplelayer(Layer):
    tag = 'simplelayer'

    '''
    初始参数
    '''
    def init_params(self):
        shape = self.inshape + self.outshape
        #print("inshape:", self.inshape)
        #print("outshape:", self.outshape)
        #print("inshape+outshape:", shape)

        n = utils.flat_shape(self.inshape)
        n *= utils.flat_shape(self.outshape)

        w_val = np.arange(n).reshape(shape).astype('float64')
        self.__W = LayerParam(self.name, 'W', w_val)

    @property
    def params(self):
        return [self.__W]

    '''
    向前传播
    in_batch: 一批输入数据
    training: 是否正在训练
    '''
    def forward(self, in_batch, training=False):
        out = in_batch @ self.__W.value
        return self.activation(out)

    #反向传播梯度
    def backward(self, gradient):
        grad = self.activation.grad(gradient)
        self.__W.gradient = self.__W.value

        res = grad @ self.__W.value.T
        return res
