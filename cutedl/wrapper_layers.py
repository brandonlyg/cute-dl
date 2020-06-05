# coding=utf-8

import pdb
import numpy as np
from model import Layer

'''
包装器层实现
'''
class Wrapper(Layer):
    tag = 'Wrapper'

    '''
    merge_model 输出合并模式. 输出合并都是针对张量的最后一个维度
        'sum': 求和 a+b
        'mul': 按元素相乘 a * b
        'concat': 连接 a concat b
        'ave': 平均 (a+b)/2
    '''
    def __init__(self, merge_model):
        #所有支持的合并方式
        mms = set(['sum', 'mul', 'concat', 'ave'])
        if merge_model not in mms:
            raise Exception("can't support the merge model:"+str(merge_model))

        self.__merge_model = merge_model

        super().__init__()

        self.__shape1 = None
        self.__shape2 = None

    @property
    def merge_model(self):
        return self.__merge_model

    '''
    合并输出
    '''
    def merge_out(self, out1, out2):
        mm = self.__merge_model
        if 'concat' == mm:
            return self.__concat_merge(out1, out2)

    '''
    合并输出形状
    '''
    def merge_outshape(self, shape1, shape2):
        mm = self.__merge_model
        if 'concat' == mm:
            return self.__concat_merge_outshape(shape1, shape2)


    '''
    根据合并输出模式提取梯度
    '''
    def extract_gradient(self, grad):
        mm = self.__merge_model
        if 'concat' == mm:
            return self.__concat_extract_grad(grad)

    '''
    按concat方式和并输出，提取梯度
    '''
    def __concat_merge(self, out1, out2):
        shape1 = out1.shape
        shape2 = out2.shape

        if shape1[:-1] != shape2[:-1]:
            raise Exception("can't concat data with shape: %s, %s"%(str(shape1), str(shape2)))

        n1 = shape1[-1]
        n2 = shape2[-1]

        shape = shape1[:-1] + (n1+n2,)
        out = np.zeros(shape)
        out[:,:,0:n1] = out1
        out[:,:,n1:n1+n2] = out2

        self.__shape1 = shape1
        self.__shape2 = shape2

        return out

    def __concat_merge_outshape(self, shape1, shape2):
        shape = shape1[:-1]
        n1 = shape1[-1]
        n2 = shape2[-1]
        shape = shape + (n1+n2,)
        return shape

    def __concat_extract_grad(self, grad):
        shape1 = self.__shape1
        shape2 = self.__shape2

        n1 = shape1[-1]
        n2 = shape2[-1]

        grad1 = grad[:,:,0:n1]
        grad2 = grad[:,:,n1:n1+n2]

        return grad1, grad2

'''
双向层
输入数据必须是3D张量
'''
class Bidirectional(Wrapper):
    tag = "Bidiectional"

    '''
    layer 正向层
    backward_layer 反向层

    正向层和反向层的输入形状必须一样
    '''
    def __init__(self, layer, backward_layer, merge_model='concat'):
        self.__layer = layer
        self.__backward_layer = backward_layer

        if merge_model is None:
            merge_model = 'concat'

        super().__init__(merge_model)


    def set_parent(self, parent):
        super().set_parent(parent)

        self.__layer.set_parent(self)
        self.__backward_layer.set_parent(self)

    def init_params(self):
        self.__layer.init_params()
        self.__backward_layer.init_params()

    @property
    def params(self):
        params = self.__layer.params
        params = params + self.__backward_layer.params
        return params

    @property
    def inshape(self):
        return self.__layer.inshape

    @property
    def outshape(self):
        return self.merge_outshape(self.__layer.outshape, self.__backward_layer.outshape)

    def set_prev(self, layer):
        super().set_prev(layer)

        self.__layer.set_prev(layer)
        self.__backward_layer.set_prev(layer)


    def forward(self, in_batch, training):
        m, k, n = in_batch.shape
        #倒序第二个维度
        bk_in_batch = in_batch[:, range(k-1, -1, -1)]

        out = self.__layer.forward(in_batch, training)
        bk_out = self.__backward_layer.forward(bk_in_batch, training)

        out = self.merge_out(out, bk_out)
        return out

    def backward(self, gradient):
        grad, bk_grad = self.extract_gradient(gradient)

        grad = self.__layer.backward(grad)
        bk_grad = self.__backward_layer.backward(bk_grad)

        m, k, n  = gradient.shape
        #恢复倒序的梯度
        bk_grad = bk_grad[:, range(k-1, -1, -1)]

        grad = grad + bk_grad
        return grad

    def reset(self):
        self.__layer.reset()
        self.__backward_layer.reset()
