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

    def __init__(self, outshape, inshape=None, activation='relu',
                weight_initializer='uniform', bias_initializer='zeros'):
        #输出形状(int,)
        if type(outshape) != type(1):
            raise Exception("invalid outshape: "+str(outshape))

        self.__outshape = self.check_shape(outshape)

        #输入形状
        self.__inshape = None
        if inshape is not None:
            #pdb.set_trace()
            if type(inshape) != type(1):
                raise Exception("invalid inshape: "+str(inshape))

            self.__inshape = self.check_shape(inshape)

        #print("Dense kargs:", kargs)
        #参数
        self.__W = self.weight_initializers[weight_initializer]
        self.__b = self.bias_initializers[bias_initializer]

        super().__init__(activation)

        #输入数据, 已经转换成合适的形状
        self.__in_batch = None

    def init_params(self):

        #展平纬度, 初始化参数值
        shape = self.__inshape + self.__outshape
        #pdb.set_trace()
        wval = self.__W(shape)
        bval = self.__b((shape[1],))

        self.__W = LayerParam(self.name, 'weight', wval)
        self.__b = LayerParam(self.name, 'bias', bval)

    @property
    def params(self):
        return [self.__W, self.__b]

    def set_prev(self, prev_layer):
        self.__inshape = self.check_shape(prev_layer.outshape)
        #pdb.set_trace()
        super().set_prev(prev_layer)

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    def forward(self, in_batch, training=False):
        W = self.__W.value
        b = self.__b.value

        out = in_batch @ W + b

        self.__in_batch = in_batch

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

        return out_grad

    #重置当前层的状态
    def reset(self):
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
    def __init__(self, keep_prob):
        self.__keep_prob = keep_prob

        super().__init__()
        #pdb.set_trace()
        self.__mark = None

    def init_params(self):
        pass

    @property
    def params(self):
        return []

    @property
    def inshape(self):
        return self.prev.outshape

    @property
    def outshape(self):
        return self.inshape

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


'''
flatten 展平层
'''
class Flatten(Layer):

    def __init__(self):
        self.__inshape = None
        self.__outshape = None

        super().__init__()

    def set_prev(self, prev_layer):
        inshape = prev_layer.outshape
        self.__inshape = inshape
        self.__outshape = (utils.flat_shape(inshape), )

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    def forward(self, in_batch, training=False):
        return in_batch.reshape((-1,)+self.__outshape)

    def backward(self, gradient):
        return gradient.reshape((-1,)+self.__inshape)


'''
过滤层,
把三个维度的张量过滤成两个维度的张量
eg (m, k, n) --> (m, n)
'''
class Filter(Layer):

    def __init__(self):
        self.__inshape = None
        self.__outshape = None

        super().__init__()

        self.__in_batch_shape = None

    def set_prev(self, prev_layer):
        inshape = (prev_layer.outshape[-1], )
        self.__inshape = inshape
        self.__outshape = inshape

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    def forward(self, in_batch, training=False):
        self.__in_batch_shape = in_batch.shape

        out = in_batch[:, -1, :]

        return out

    def backward(self, gradient):
        out = np.zeros(self.__in_batch_shape)
        out[:, -1, :] = gradient

        return out

    def reset(self):
        self.__in_batch_shape = None

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
