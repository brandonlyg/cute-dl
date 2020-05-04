# coding=utf-8

import os
import pdb
import pickle
import numpy as np

import activations
import reguls

'''
层参数类型
'''
class LayerParam(object):

    '''
    layer_name: 所属层的的名字
    name: 参数名
    value: 参数值
    '''
    def __init__(self, layer_name, name, value):
        self.__name = layer_name+"/"+name
        self.value = value

        #梯度
        self.gradient = None
        #更新次数
        self.udt = 0

    @property
    def name(self):
        return self.__name

    @classmethod
    def reset(cls, p):
        layer_name = os.path.dirname(p.name)
        name = p.name[len(layer_name)+1:]
        new_p = LayerParam(p.name, name, p.value)
        return new_p

'''
层的抽象类型
'''
class Layer(object):
    tag = 'Layer'

    '''
    activation: 激活函数的名字
    '''
    def __init__(self, activation='linear'):

        #得到激活函数
        self.__activation = activations.get(activation)

        #层在模型中的id, 是层在模型中的索引
        self.__id = 0
        #层的名字
        self.__name = '/%d-%s'%(self.__id, self.tag)

        #上一个层
        self.__prev = None
        #下一个层
        self.__next = None

    '''
    检查形状
    '''
    def check_shape(self, shape):
        if type(shape) == type(1):
            return (shape, )
        elif type(shape) == type((1, 2)):
            return shape
        else:
            return None

    @property
    def layer_id(self):
        return self.__id

    '''
    层的名字
    '''
    @property
    def name(self):
        return self.__name

    '''
    激活函数
    '''
    @property
    def activation(self):
        return self.__activation

    '''
    初始参数
    '''
    def init_params(self):
         raise Exception("the init_params method not implement!")

    '''
    返回参数列表: [LayerParam,...]
    '''
    @property
    def params(self):
        raise Exception("the params method not implement!")

    '''
    上一个层属性
    '''
    @property
    def prev(self):
        return self.__prev

    '''
    当前层要根据上一个层outshape确定inshape
    '''
    def set_prev(self, prev_layer):
        self.__prev = prev_layer
        self.__id = prev_layer.layer_id + 1
        self.__name = '/%d-%s'%(self.__id, self.tag)

    '''
    下一个层
    '''
    @property
    def next(self):
        return self.__next

    '''
    要确保当前层的outshape和下个层的的inshape具有相同的维度
    当前层的set_prev会在set_next之前调用
    '''
    def set_next(self, next_layer):
        #pdb.set_trace()
        if len(next_layer.inshape) != len(self.outshape):
            raise Exception("the layer %s outshape %s and the next layer %s inshape %s. Dimension not match"%(
                            self.name, str(self.outshape),
                            next_layer.name, str(next_layer.inshape)
                            ))

        self.__next = next_layer

    '''
    输入形状
    无论什么时候都应返回一个tuple对象, 通过这个对象可以知道inshape的维度。
    除第一个层外, 其他层的inshape对象在调用prev.setter之后才能确定。
    '''
    @property
    def inshape(self):
        raise Exception("the inshape property not implement!")

    '''
    输出形状
    调用next.setter时, 当前层应根据next_layer的inshape维度调整自己的的outshape
    '''
    @property
    def outshape(self):
        raise Exception("the outshape property not implement!")

    '''
    向前传播
    in_batch: 一批输入数据
    training: 是否正在训练
    '''
    def forward(self, in_batch, training=False):
        raise Exception("the params method not implement!")

    #反向传播梯度
    def backward(self, gradient):
        raise Exception("the params method not implement!")

    #重置当前层的状态
    def reset(self):
        pass

'''
模型类
'''
class Model(object):

    '''
    layers: Layer list
    '''
    def __init__(self, layers=None):
        self.__layers = layers

    def __check(self):
        if self.__layers is None or len(self.__layers) == 0:
            raise Exception("layers is None")

    '''
    添加层
    layer: Layer类型的对象
    '''
    def add(self, layer):
        if self.__layers is None:
            self.__layers = []

        self.__layers.append(layer)

        return self

    '''
    得到一个Layer对象
    idx: Layer对象的索引
    '''
    def get_layer(self, index):
        self.__check()
        if len(self.__layers) <= index:
            raise Exception("index out of range %d"%len(self.__layers))

        return self.__layers[index]

    @property
    def layer_count(self):
        return len(self.__layers)

    '''
    得到层的迭代器
    '''
    def layer_iterator(self):
        self.__check()

        for ly in self.__layers:
            yield ly

    '''
    组装模型
    '''
    def assemble(self):
        self.__check()
        count = len(self.__layers)

        #把层按顺序组装起来
        pre_ly = None
        for ly in self.__layers:
            if pre_ly is None:
                pre_ly = ly
                continue

            pre_ly.set_next(ly)
            ly.set_prev(pre_ly)

            #上一个层的前、后层已经已经确定, 初始化参数
            pre_ly.init_params()

            pre_ly = ly

        if pre_ly is not None:
            pre_ly.init_params()

    '''
    打印模型信息摘要
    '''
    def summary(self):
        pass

    '''
    使用模型预测
    in_batch: 一批输入数据
    '''
    def predict(self, in_batch, training=False):
        self.__check()

        out = in_batch
        for ly in self.__layers:
            out = ly.forward(out, training)

        return out

    '''
    反向传播梯度
    '''
    def backward(self, gradient):
        g = gradient
        #pdb.set_trace()
        count = len(self.__layers)
        for i in range(count-1, -1, -1):
            ly = self.__layers[i]
            g = ly.backward(g)

    '''
    重置所有层的状态
    '''
    def reset(self):
        for ly in self.__layers:
            ly.reset()

    '''
    保存模型
    '''
    def save(self, fpath):
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.mkdir(dir)

        self.reset()
        realfp = fpath + ".m.pkl"
        with open(realfp, 'wb') as f:
            pickle.dump(self, f)

    '''
    加载模型
    '''
    @classmethod
    def load(cls, fpath):
        realfp = fpath + ".m.pkl"
        if not os.path.exists(realfp):
            return None

        model = None
        with open(realfp, 'rb') as f:
            model = pickle.load(f)

        return model
