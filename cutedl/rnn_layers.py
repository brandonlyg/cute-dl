# coding=utf-8

'''
rnn 层实现
'''

import pdb
import numpy as np
from model import Layer, LayerParam
import activations

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

        self.__hs= None
        self.__in_batchs = None


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

        if training:
            if self.__hs is None:
                self.__hs = []
            if self.__in_batchs is None:
                self.__in_batchs = []
            self.__hstack.append(hs)
            self.__in_batchs.append(in_batch)

        return self.activation(out)

    '''
    return grad_in_batch, grad_hs
    '''
    def backward(self, gradient):
        grad = self.activation.grad(gradient)

        W = self.__W.value
        Wh = self.__Wh.value
        pre_hs = self.__hs.pop()
        in_batch = self.__in_batchs.pop()

        grad_in_batch = grad @ W.T
        self.__W.gradient = in_batch.T @ grad

        grad_hs = grad @ Wh.T
        self.__Wh.gradient = pre_hs.T @ grad

        self.__b.gradient = grad.sum(axis=0)

        return grad_in_batch, grad_hs

    def reset(self):
        self.__in_batchs = None
        self.__hs = None

        self.__W = LayerParam.reset(self.__W)
        self.__Wh = LayerParam.reset(self.__Wh)
        self.__b = LayerParam.reset(self.__b)

'''
记忆单元
'''
class MemoryUnit:

    def __init__(self, activation='tanh'):
        self.__activation = activations.get(activation)

        self.__memories = None
        self.__inputs = None
        self.__mcs = None

        self.__pre_memory = None


    def forward(self, forget, input, memory_choice, training):
        if self.__pre_memory is None:
            self.__pre_memory = np.zeros(forget.shape)

        cur_m = forget * self.__pre_memory + input * memory_choice

        if training:
            if self.__memories is None:
                self.__memories = []
            if self.__inputs is None:
                self.__inputs = []
            if self.__mcs is None:
                self.__mcs = []

            self.__memories.append(self.__pre_memory)
            self.__inputs.append(input)
            self.__mcs.append(memory_choice)

        self.__pre_memory = cur_m

        return self.__activation(cur_m)

    def backward(self, gradient):
        grad = self.__activation.grad(gradient)

        pre_m = self.__memories.pop()
        input = self.__inputs.pop()
        mc = self.__mcs.pop()

        grad_forget = grad * pre_m
        grad_input = grad * mc
        grad_mc = grad * input

        return grad_forget, grad_input, grad_mc

    def reset(self):
        self.__memories = None
        self.__inputs = None
        self.__mcs = None

        self.__pre_memory = None

'''
输出单元
'''
class OutUnit:
    def __init__(self):
        self.__outs = None
        self.__memories = None

    def forward(self, out, memory, training):
        res = out * memory

        if training:
            if self.__outs is None:
                self.__outs = []
            if self.__memories is None:
                self.__memories = []
            self.__outs.append(out)
            self.__memories.append(memory)

        return res

    def backward(self, gradient):
        out = self.__outs.pop()
        memory = self.__memories.pop()

        grad_out = gradient * memory
        grad_memory = gradient * out

        return grad_out, grad_memory

    def reset(self):
        self.__outs = None
        self.__memories = None

'''
Long Shot-Term Memory Layer
'''
class LSTM(Layer):

    '''
    units 输出维度
    '''
    def __int__(self, out_units, in_units=None):
        if type(out_units) != type(1):
            raise Exception("invalid out_units: "%str(out_units))

        self.__out_units = out_units

        if in_units is not None:
            if type(in_units) != type(1):
                raise Exception("invalid in_units: "%str(in_units))
            self.__in_units = in_units
            self.__init_gates()

        super().__init__()

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

        #记忆单元
        self.__memory_unit = MemoryUnit()
        #输出单元
        self.__out_unit = OutUnit()


    @property
    def inshape(self):
        return (self.__in_units, )

    @property
    def outshape(self):
        return (self.__out_units, )

    @property
    def params(self):
        res = self.__g_in.params
        res += self.__g_forget.params
        res += self.__g_out.params
        res += self.__g_memory.params
        return res

    def init_params(self):
        pass

    def set_prev(self, prev_layer):
        outshape = prev_layer.outshape
        self.__in_units = outshape[-1]
        self.__init_gates()
        super().set_prev(prev_layer)

    def __lstm_forward(self, in_batch, hs, training):
        g_in = self.__g_in.forward(in_batch, hs, training)
        g_forget = self.__g_forget.forward(in_batch, hs, training)
        g_out = self.__g_out.forward(in_batch, hs, training)
        g_memory = self.__g_memory.forward(in_batch, hs, training)

        memory = self.__memory_unit.forward(g_forget, g_in, g_memory, training)
        cur_hs = self.__out_unit.forward(g_out, memory)

        return cur_hs

    def __lstm_backward(self, gradient):
        grad_out, grad_memory = self.__out_unit.backward(gradient)
        grad_forget, grad_in, grad_gm = self.__memory_unit.backward(grad_memory)

        grad_in_batch, grad_hs = self.__g_memory.backward(grad_gm)
        tmp1, tmp2 = self.__g_out.backward(grad_out)
        grad_in_batch += tmp1
        grad_hs += tmp2

        tmp1, tmp2 = self.__g_forget.backward(grad_forget)
        grad_in_batch += tmp1
        grad_hs += tmp2

        tmp1, tmp2 = self.__g_forget.backward(grad_in)
        grad_in_batch += tmp1
        grad_hs += tmp2

        return grad_in_batch, grad_hs

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
        self.__g_in.reset()
        self.__g_forget.reset()
        self.__g_out.reset()
        self.__g_memory.reset()

        self.__memory_unit.reset()
        self.__out_unit.reset()
        
