# coding=utf-8

import pdb

import numpy as np
from cutedl.model import Layer, LayerParam
from cutedl import utils

'''
测试用例中使用的layer
'''

class Simplelayer(Layer):
    tag = 'Simplelayer'

    def __init__(self, outshape, inshape=None):
        self.__outshape = self.check_shape(outshape)
        self.__inshape = (-1,)

        self.__W = None

        if inshape is not None:
            self.__inshape = self.check_shape(inshape)
            if self.__inshape is None:
                raise Exception("invalid inshape: %s"%str(inshape))

        super().__init__()

    '''
    初始参数
    '''
    def init_params(self):
        shape = self.inshape + self.outshape
        #print("inshape:", self.inshape)
        #print("outshape:", self.outshape)
        print("inshape+outshape:", shape)

        n = utils.flat_shape(self.inshape)
        n *= utils.flat_shape(self.outshape)

        w_val = np.arange(n).reshape(shape).astype('float64')
        self.__W = LayerParam(self.name, 'weight', w_val)

    @property
    def params(self):
        return [self.__W]

    def set_prev(self, prev_layer):
        #pdb.set_trace()
        self.__inshape = prev_layer.outshape
        super().set_prev(prev_layer)

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    '''
    向前传播
    in_batch: 一批输入数据
    training: 是否正在训练
    '''
    def forward(self, in_batch, training=False):
        #pdb.set_trace()
        out = in_batch @ self.__W.value
        return self.activation(out)

    #反向传播梯度
    def backward(self, gradient):
        grad = self.activation.grad(gradient)
        self.__W.gradient = self.__W.value
        #pdb.set_trace()
        res = grad @ self.__W.value.T
        return res
