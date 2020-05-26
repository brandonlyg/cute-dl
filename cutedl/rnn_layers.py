# coding=utf-8

'''
rnn 层实现
'''

import pdb
import numpy as np
from model import Layer, LayerParam
import activations

'''
Embedding 嵌入层实现
这个层只能是输入层
'''
class Embedding(Layer):
    tag='Embedding'

    '''
    dims 嵌入向量维度
    vocabulary_size 词汇表大小
    need_train 是否需要训练嵌入向量
    '''
    def __init__(self, dims, vocabulary_size, need_train=True):
        #初始化嵌入向量
        initializer = self.weight_initializers['uniform']
        self.__vecs = initializer((vocabulary_size, dims))

        super().__init__()

        self.__params = None
        if need_train:
            self.__params = []
            self.__cur_params = None
            self.__in_batch = None

    def init_params(self):
        if self.__params is None:
            return

        voc_size, _ = self.__vecs.shape
        for i in range(voc_size):
            pname = 'weight_%d'%i
            p = LayerParam(self.name, pname, self.__vecs[i])
            self.__params.append(p)


    @property
    def params(self):
        return self.__cur_params

    @property
    def inshape(self):
        return (-1, -1)

    @property
    def outshape(self):
        return (-1, -1, self.__vecs.shape[-1])

    @property
    def vectors(self):
        return self.__vecs

    '''
    in_batch shape=(m, T)
    return shape (m, T, dims)
    '''
    def forward(self, in_batch, training):
        m,T = in_batch.shape
        outshape = (m, T, self.outshape[-1])
        out = np.zeros(outshape)

        for i in range(m):
            out[i] = self.__vecs[in_batch[i]]

        if training and self.__params is not None:
            self.__in_batch = in_batch

        return out

    def backward(self, gradient):
        if self.__params is None:
            return

        #pdb.set_trace()
        in_batch = self.__in_batch
        params = {}
        m, T, _ = gradient.shape
        for i in range(m):
            for t in range(T):
                grad = gradient[i, t]
                idx = self.__in_batch[i, t]
                if idx not in params:
                    params[idx] = self.__params[idx]
                p = params[idx]
                if p.gradient is None:
                    p.gradient = grad
                else:
                    p.gradient += grad

        self.__cur_params = list(params.values())
        #print("params keys: ", list(params.keys()))

    def reset(self):
        self.__cur_params = None

        if self.__params is not None:
            count = len(self.__params)
            for i in range(count):
                p = self.__params[i]
                self.__params[i] = LayerParam.reset(p)


'''
RNN 层定义
'''
class RNN(Layer):
    tag='RNN'

    def __init__(self, out_units, in_units=None, activation='linear'):
        if type(out_units) != type(1):
            raise Exception("invalid out_units: "+str(out_units))

        self.__out_units = out_units

        if in_units is not None:
            if type(in_units) != type(1):
                raise Exception("invalid in_units: "+str(in_units))

            self.__in_units = in_units

        super().__init__(activation)

    @property
    def in_units(self):
        return self.__in_units

    @property
    def out_units(self):
        return self.__out_units

    @property
    def inshape(self):
        return (-1, -1, self.__in_units)

    @property
    def outshape(self):
        return (-1, -1, self.__out_units)

    def set_prev(self, prev_layer):
        outshape = prev_layer.outshape
        self.__in_units = outshape[-1]
        super().set_prev(prev_layer)

    def hiden_forward(self, in_batch, pre_hs, training):
        raise Exception("hiden_forward not implement!")

    def hiden_backward(self, gradient):
        raise Exception("hiden_backward not implement!")

    def forward(self, in_batch, training):
        m, T, n = in_batch.shape

        hstatus = np.zeros(in_batch.shape)
        pre_hs = np.zeors((m, n))
        for t in range(T):
            hstauts[:, t, :] = self.hiden_forward(in_batch[:,t,:], pre_hs, training)
            pre_hs = hstauts[:, t, :]

        return hstatus

    def backward(self, gradient):
        m, T, n = gradient.shape

        grad_x = np.zeros(gradient.shape)
        for t in range(T-1, -1, -1):
            grad_x[:,t,:], grad_hs = self.hiden_backward(gradient[:,t,:])
            if t - 1 >= 0:
                gradient[:,t-1,:] += grad_hs

        return grad_x


'''
隐藏门单元
'''
class GateUnit(Layer):
    tag = 'RNN-GateUnit'

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

        #pdb.set_trace()
        super().__init__(activation, parent_layer, layer_id)

        self.init_params()

        self.__hs = None
        self.__in_batchs = None


    def init_params(self):
        initializers = self.bias_initializers

        shape = self.__inshape + self.__outshape
        val = initializers['uniform'](shape)
        self.__W = LayerParam(self.name, 'weight', val)

        shape = self.__outshape + self.__outshape
        val = initializers['uniform'](shape)
        self.__Wh = LayerParam(self.name, 'weight_hiden', val)

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
            self.__hs.append(hs)
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
乘法单元
'''
class MultiplyUnit:

    def __init__(self):
        pass

    def forward(self, left, right, training):
        out = left * right
        if training:
            self.__lefts.append(left)
            self.__rights.append(right)

        return out

    def backward(self, gradient):
        left = self.__lefts.pop()
        right = self.__rights.pop()

        grad_left = gradient * right
        grad_right = gradient * left

        return grad_lef, grad_right

    def reset(self):
        self.__lefts = []
        self.__lefts = []

'''
GRU-RNN 层输出单元
'''
class GRUOutUnit:

    def __int__(self):
        self.__gus = []
        self.__pre_hs = []
        self.__cddouts = []

    def forward(self, gu, pre_hs, cddout, training):
        out = gu * pre_hs + (1-gu) * cddout

        if training:
            self.__gus.append(gu)
            self.__pre_hs.append(pre_hs)
            self.__cddouts.append(cddouts)

        return out

    def backward(self, gradient):
        gu = self.__gus.pop()
        pre_hs = self.__pre_hs.pop()
        cddout = self.__cddouts.pop()

        grad_gu = gradient * (pre_hs - cddout)
        grad_pre_hs = gradient * gu
        grad_cddout = gradient * (1-gu)

        return grad_gu, grad_pre_hs, grad_cddout


    def reset(self):
        self.__gus = []
        self.__pre_hs = []
        self.__cddouts = []

'''
Gate Recurrent Units
'''
class GRU(RNN):
    tag='RNN-GRU'

    def __init__(self, out_units, in_units=None, activation='linear'):
        #重置门
        self.__g_reset = None
        #更新门
        self.__g_update = None
        #候选输出门
        self.__g_cddout = None

        #重置门乘法单元
        self.__u_gr = None
        #输出单元
        self.__u_out = None

        super().__init__(out_units, in_units, activation=activation)

    def init_params(self):
        out_units = self.out_units
        in_units = self.in_units

        #pdb.set_trace()
        self.__g_reset = GateUnit(out_units, in_units, parent_layer=self, layer_id=1)
        self.__g_update = GateUnit(out_units, in_units, parent_layer=self, layer_id=2)
        self.__g_cddout = GateUnit(out_units, in_units, parent_layer=self, layer_id=3)

        self.__u_gr = MultiplyUnit()
        self.__u_out = GRUOutUnit()

    @property
    def params(self):
        res = self.__g_reset.params
        res += self.__g_update.params
        res += self.__g_cddout.params
        return res

    def hiden_forward(self, in_batch, pre_hs, training):
        gr = self.__g_reset.forward(in_batch, pre_hs, training)
        gu = self.__g_udate.forward(in_batch, pre_hs, training)
        ugr = self.__u_gr.forward(gr, pre_hs)
        cddo = self.__g_cddout.forward(inbatch, ugr)

        hs = self.__u_out.forward(gu, pre_hs, cddo)

        return hs

    def hiden_backward(self, gradient):
        grad_gu, grad_pre_hs, grad_cddo = self.__u_out.backward(gradient)
        grad_in_batch, grad_ugr = self.__g_cddout.backward(grad_cddu)

        grad_gr, g_pre_hs = self.__u_gr.backward(grad_ugr)
        grad_pre_hs = grad_pre_hs + g_pre_hs

        g_in_batch, g_pre_hs = self._g_update.backward(grad_gu)
        grad_in_batch = grad_in_batch + g_in_batch
        grad_pre_hs = grad_pre_hs + g_pre_hs

        g_in_batch, g_pre_hs = self._g_reset.backward(grad_gr)
        grad_in_batch = grad_in_batch + g_in_batch
        grad_pre_hs = grad_pre_hs + g_pre_hs

        return grad_in_batch, grad_pre_hs

    def reset(self):
        self.__g_update.reset()
        self.__g_reset.reset()
        self.__u_gr.reset()
        self.__g_cddout.reset()
        self.__u_out.reset()

'''
LSTM 记忆单元
'''
class LSTMMemoryUnit:

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
LSTM 输出单元
'''
class LSTMOutUnit:
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
class LSTM(RNN):
    tag='RNN-LSTM'

    '''
    units 输出维度
    '''
    def __int__(self, out_units, in_units=None, activation='linear'):
        super().__init__(out_units, in_units, activation)

    '''
    初始化门
    '''
    def init_params(self):
        in_units = self.in_units
        out_units = self.out_units

        #输入门
        self.__g_in = GateUnit(out_units, in_units, parent_layer=self, layer_id=1)
        #遗忘门
        self.__g_forget = GateUnit(out_units, in_units, parent_layer=self, layer_id=2)
        #输入门
        self.__g_out = GateUnit(out_units, in_units, parent_layer=self, layer_id=3)
        #记忆门
        self.__g_memory = GateUnit(out_units, in_units, activation='tanh', parent_layer=self, layer_id=4)

        #记忆单元
        self.__memory_unit =LSTMMemoryUnit()
        #输出单元
        self.__out_unit = LSTMOutUnit()


    @property
    def params(self):
        res = self.__g_in.params
        res += self.__g_forget.params
        res += self.__g_out.params
        res += self.__g_memory.params
        return res



    def hiden_forward(self, in_batch, hs, training):
        g_in = self.__g_in.forward(in_batch, hs, training)
        g_forget = self.__g_forget.forward(in_batch, hs, training)
        g_out = self.__g_out.forward(in_batch, hs, training)
        g_memory = self.__g_memory.forward(in_batch, hs, training)

        memory = self.__memory_unit.forward(g_forget, g_in, g_memory, training)
        cur_hs = self.__out_unit.forward(g_out, memory)

        return cur_hs

    def hiden_backward(self, gradient):
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

    def reset(self):
        self.__g_in.reset()
        self.__g_forget.reset()
        self.__g_out.reset()
        self.__g_memory.reset()

        self.__memory_unit.reset()
        self.__out_unit.reset()
