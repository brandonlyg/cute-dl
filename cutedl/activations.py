# coding=utf-8

import dlmath

'''
激活函数
'''
class Activation(object):
    name=''

    def __call__(self, in_batch):
        raise Exception("__call__ not implement")

    '''
    求梯度
    gradient: 该函数输出值的梯度
    '''
    def grad(self, gradient):
        raise Exception("gradient not implement")

'''
线性激活函数, 没有激活
'''
class Linear(Activation):
    name='linear'

    def __call__(self, in_batch):
        return in_batch

    def grad(self, gradient):
        return gradient

'''
relu 激活函数
'''
class Relu(Activation):
    name='relu'

    def __init__(self):
        self.__grad = None

    def __call__(self, in_batch):
        #得到 <= 0的数据的索引
        indices =  in_batch <= 0

        in_batch[indices] = 0
        self.__grad = indices

        return in_batch

    def grad(self, gradient):
        gradient[self.__grad] = 0
        self.__grad = None
        return gradient


'''
sigmoid 激活函数
'''
class Sigmoid(Activation):
    name = 'sigmoid'

    def __init__(self):
        self.__grad = None

    def __call__(self, in_batch):
        out = dlmath.sigmoid(in_batch)
        self.__grad = out*(1-out)
        return out

    def grad(self, gradient):
        return gradient * self.__grad



'''
tanh 激活函数
'''
class Tanh(Activation):
    name = 'tanh'

    def __init__(self):
        self.__grad = None

    def __call__(self, in_batch):
        out = dlmath.tanh(in_batch)
        self.__grad = 1 - out**2
        return out

    def grad(self, gradient):
        return gradient * self.__grad


act_dict = {
    Linear.name: Linear,
    Relu.name: Relu,
    Sigmoid.name: Sigmoid,
    Tanh.name: Tanh
}

#创建激活函数
def get(name):
    #print(act_dict)
    #print('name:', name)
    ACT = act_dict[name]
    return ACT()
