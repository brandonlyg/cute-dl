# coding=utf-8

import pdb

import numpy as np
from model import Layer, LayerParam
import utils
import dlmath

'''
同样神经网络层
'''

'''
全连接层
'''
class Dense(Layer):
    tag='Dense'

    def __init__(self, outshape, inshape=None, activation='relu'):
        #输出形状(int,)
        if type(outshape) != type(1):
            raise Exception("invalid outshape: "+str(outshape))

        self.__outshape = self.check_shape(outshape)

        #输入形状
        self.__inshape = None
        if inshape is not None:
            if type(inshape) != type(1):
                raise Exception("invalid inshape: "+str(inshape))

            self.__in_shape = self.check_shape(inshape)

        #print("Dense kargs:", kargs)
        #参数
        self.__W = None
        self.__b = None

        super().__init__(activation)

        #输入数据的原始形状
        self.__in_batch_shape = None
        #输入数据, 已经转换成合适的形状
        self.__in_batch = None

    def init_params(self):

        #展平纬度, 初始化参数值
        std = 0.01
        shape = self.__inshape + self.__outshape
        wval = np.random.randn(shape[0], shape[1]) * std
        bval = np.zeros(shape[1])

        self.__W = LayerParam(self.name, 'weight', wval)
        self.__b = LayerParam(self.name, 'bias', bval)

    @property
    def params(self):
        return [self.__W, self.__b]

    def join(self, pre_layer):
        #只取最后一个维度
        inshape = pre_layer.outshape[-1]
        self.__inshape = self.check_shape(inshape)

        super().join(pre_layer)

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    def forward(self, in_batch, training=False):
        W = self.__W.value
        b = self.__b.value
        self.__in_batch_shape = in_batch.shape

        in_x = in_batch
        if len(in_batch.shape) > 2:
            #超过两个维度把数据转换成(m*k,n), 只取[m*(k-1):, :]
            m = in_batch.shape[0]
            n = in_batch.shape[-1]
            k = utils.flat_shape(in_batch.shape)//(m*n)
            in_x = in_batch.reshape((m*k, n))[m*(k-1):, :]

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
        #数据梯度 (m,inshape) = (m,outshape) @ (outshape, inshape)
        out_grad = grad @ W.T

        if len(self.__in_batch_shape) > 2:
            #还原成输入数据的形状
            m = self.__in_batch_shape[0]
            n = self.__in_batch_shape[1]
            k = utils.flat_shape(self.__in_batch_shape)//(m*n)
            tmp = np.zeros(self.__in_batch_shape).reshape(m*k, n)
            tmp[m*(k-1):, :] = out_grad
            out_grad = tmp.reshape(self.__in_batch_shape)

        return out_grad

    #重置当前层的状态
    def reset(self):
        self.__in_batch_shape = None
        self.__in_batch = None

        self.__W = LayerParam.reset(self.__W)
        self.__b = LayerParam.reset(self.__b)



'''
dropout层
'''
class Dropout(Layer):
    tag='Dropout'

    '''
    keep_prob: 保留概率取值区间为(0, 1]
    '''
    def __init__(self, *outshape, **kargs):
        self.__keep_prob = 1
        if 'keep_prob' in kargs:
            self.__keep_prob = kargs['keep_prob']

        super().__init__(-1, **kargs)
        #pdb.set_trace()
        self.__mark = None

    def init_params(self):
        pass

    @property
    def params(self):
        return []

    def forward(self, in_batch, training=False):
        kp = self.__keep_prob
        #pdb.set_trace()
        if not training or kp <= 0 or kp>=1:
            return in_batch

        #生成[0, 1)直接的均价分布
        tmp = np.random.uniform(size=in_batch.shape)
        #保留/丢弃索引
        mark = (tmp <= kp).astype(int)
        #丢弃数据, 并拉伸保留数据
        out = (mark * in_batch)/kp

        self.__mark = mark

        return out

    def backward(self, gradient):
        #pdb.set_trace()
        if self.__mark is None:
            return gradient

        out = (self.__mark * gradient)/self.__keep_prob

        return out

    def reset(self):
        self.__mark = None

#
# '''
# softmax层
# '''
# class Softmax(Layer):
#     tag='softmax'
#
#     def init_params(self):
#         pass
#
#     @property
#     def params(self):
#         return []
#
#     def forward(self, in_batch, training=False):
#         if training: #如果是训练阶段, 不做任何处理
#             return training
#
#         out, _ = utils.reduce(in_batch)
#         out = np.argmax(out, axis=1)
#         return out
#
#     def backward(self, gradient):
#         return gradient
#
#     def reset(self):
#         pass
