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

    def reset(self):
        self.gradient = None
        self.udt = 0

'''
层的抽象类型
'''
class Layer(object):
    tag = 'Layer'

    '''
    outshape: 输出形状 2 或者 (2,3)
    kargs:
        activation: 激活函数的名字
        inshape: 输入形状
    '''
    def __init__(self, *outshape, **kargs):
        #输出形状
        if len(outshape) == 1 and type(outshape[0]) == type(()):
            self.__outshape = outshape[0]
        else:
            self.__outshape = outshape

        #输入形状
        self.__inshape = None

        #得到激活函数
        self.__activation = activations.get('linear')

        #层在模型中的id, 是层在模型中的索引
        self.__id = 0
        #层的名字
        self.__name = '/%d-%s'%(self.__id, self.tag)

        #得到可选参数
        #print("Layer kargs:", kargs)
        if 'inshape' in kargs:
            self.__inshape = kargs['inshape']
            if type(self.__inshape) != type(()):
                self.__inshape = (self.__inshape,)
            #print("------inshape:", self.__inshape)

        if 'activation' in kargs:
            self.__activation = activations.get(kargs['activation'])


        if self.__inshape is not None:
            self.init_params()

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

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    @property
    def activation(self):
        return self.__activation

    '''
    加入到模型中
    pre_layer: 前一个层
    *inshape: 输入形状
    '''
    def join(self, pre_layer, *inshape):
        self.__inshape = pre_layer.outshape
        if len(inshape) != 0:
            self.__inshape = inshape

        if self.__outshape == (-1,):
            self.__outshape = self.__inshape

        self.__id = pre_layer.layer_id + 1
        self.__name = '/%d-%s'%(self.__id, self.tag)

        self.init_params()


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

        #输入层必须要有输入形状
        ly_0 = self.__layers[0]
        if ly_0.inshape is None or len(ly_0.inshape) == 0:
            raise Exception("input layer miss inshape")

        #把每一层的输入形状设置为上一层的输出形状,
        #设置输入形状的同时, 要求该层自动初始化参数(如果有参数的话)
        pre_ly = ly_0
        for ly in self.__layers[1:]:
            ly.join(pre_ly)
            pre_ly = ly

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
