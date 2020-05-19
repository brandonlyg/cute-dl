# coding=utf-8

'''
rnn 层实现
'''

import pdb
import numpy as np
from model import Layer, LayerParam

'''
隐藏门单元
'''
class GateUnit(Layer):

    '''
    输入输入
    '''
    def __init__(self, outshape, inshape, parent_layer, layer_id, activation='sigmoid'):
        if type(outshape) != type(1):
            raise Exception("invalid outshape: %s"%str(outshape))

        if type(inshape) != type(1):
            raise Exception("invalid inshape: %s"%str(inshape))

        self.__outshape = (outshape,)
        self.__inshape = (inshape,)

        self.__W = None
        self.__Wh = None
        self.__b = None

        super().__init__(activation, parent_layer, layer_id)

        self.init_params()

        self.__hstack = []
        self.__in_batch = None


    def init_params(self):
        initializers = self.bias_initializers()

        shape = self.__inshape + self.__outshape
        val = initializers['uniform'](shape)
        self.__W = LayerParam(self.name, 'weight', val)

        val = initializers['uniform'](shape)
        slef.__Wh = LayerParam(self.name, 'weight_hiden', val)

        val = initializers['zeros']((shape[1],))
        self.__b = LayerParam(self.name, 'bias', val)

    @property
    def params(self):
        return [self.__W, self.__Wh, self.__b]

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    '''
    hs 隐藏状态
    '''
    def forward(self, in_batch, hs, training):
        W = self.__W.value
        b = self.__b.value
        Wh = self.__Wh.value

        out = in_batch @ W + hs @ Wh + b
        self.__hstack.append(hs)
        self.__in_batch = in_batch

        return self.activation(out)

    '''
    return grad_in_batch, grad_hs
    '''
    def backward(self, gradient):
        grad = self.activation.grad(gradient)

        W = self.__W.value
        Wh = self.__Wh.value
        pre_hs = self.__hstack.pop()

        grad_in_batch = grad @ W.T
        self.__W.gradient = self.__in_batch.T @ grad

        grad_hs = grad @ Wh.T
        self.__Wh.gradient = pre_hs.T @ grad

        self.__b.gradient = grad.sum(axis=0)

        return grad_in_batch, grad_hs

    def reset(self):
        self.__in_batch = None
        self.__hstack = None

        self.__W = LayerParam.reset(self.__W)
        self.__Wh = LayerParam.reset(self.__Wh)
        self.__b = LayerParam.reset(self.__b)


'''
记忆单元
'''
class MemoryUnit(Layer):

    def __init__(self, activation='tanh'):
        super().__init__(activation)

        self.__memories = []
        self.__inputs = []
        self.__mcs = []


    def forward(self, forget, input, memory_choice, training):
        pre_m = None
        if len(self.__memories) == 0:
            pre_m = np.zeros(forget.shape)
        else:
            pre_m = self.__memories[-1]

        cur_m = forget * pre_m + input * memory_choice
        self.__memories.append(cur_m)
        self.__inputs.append(inputs)
        self.__mcs.append(memory_choice)

        return self.activation(cur_m)

    def backward(self, gradient):
        grad = self.activation.grad(gradient)

        self.__memories.pop()
        pre_m = self.__memories[-1]
        input = self.__inputs.pop()
        mc = self.__mcs.pop()

        grad_forget = grad * pre_m
        grad_input = grad * mc
        grad_mc = grad * input

        return grad_forget, grad_input, grad_mc

    def reset(self):
        self.__memories = []
        self.__inputs = []
        self.__mcs = []

'''
Long Shot-Term Memory Layer
'''
class LSTM(Layer):

    '''
    units 输出维度
    '''
    def __int__(self, out_units, in_units=None):

        pass

    '''
    初始化门
    '''
    def __init_gates(self):
        in_units = self.__in_units
        out_units = self.__out_units

        #输入门
        self.__g_in = Gate(out_units, in_units, parent_layer=self, layer_id=1)
        #遗忘门
        self.__g_forget = Gate(out_units, in_units, parent_layer=self, layer_id=2)
        #输入门
        self.__g_out = Gate(out_units, in_units, parent_layer=self, layer_id=3)
        #记忆门
        self.__g_memory = Gate(out_units, in_units, activation='tanh', parent_layer=self, layer_id=4)


    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    @property
    def params(self):
        return [self.__W, self.__b]

    def init_params(self):
        pass

    def set_prev(self, prev_layer):
        pass

    '''
    in_batch shape=(m, t, n)
    '''
    def forward(self, in_batch, training=False):
        _, T, _ = in_batch.shape

        hstatus = None
        memorise = None
        pre_hs = None
        pre_memory = None
        for t in range(T):
            hstauts[:, t, :], memorise[:,t,:] = self.__lstm_forward(in_batch[:,t,:], pre_hs, pre_memory)
            pre_hs = hstauts[:, t, :]
            pre_memory = memorise[:,t,:]

        return hstatus

    def backward(self, gradient):
        _, T, _ = gradient.shape

        next_hs = None
        next_memory = None
        grad_x = None
        for t in range(T-1, -1, -1):
            grad_x[:,t,:], grad_hs = self.__lstm_backward(gradient[:,t,:])
            if t - 1 >= 0:
                gradient[:,t-1,:] = grad_hs

        return grad_x

    def reset(self):
        pass
