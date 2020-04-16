# coding=utf-8

import pdb
import numpy as np

import dlmath
import utils

'''
损失函数
'''
class Loss(object):

    '''
    梯度属性
    '''
    @property
    def gradient(self):
        raise Exception("gradient not impliment")

    '''
    计算误差和梯度
    y_true 数据的真实标签
    y_pred 模型预测的标签

    return 误差值
    '''
    def __call__(self, y_true, y_pred):
        raise Exception("__call__ not impliment")


'''
均方误差损失函数
'''
class Mse(Loss):

    def __init__(self):
        self.__grad = None

    def __call__(self, y_true, y_pred):
        err = y_pred - y_true
        loss =  (err**2).mean(axis=0)/2

        n = y_true.shape[0]
        self.__grad = err/n
        #pdb.set_trace()
        return loss.sum()

    @property
    def gradient(self):
        return self.__grad


'''
二分类交叉熵损失函数
'''
class BinaryCrossentropy(Loss):

    '''
    form_logits: 是否把输入数据转换成概率形式. 默认True. 如果输入数据已经是概率形式可以把这个参数设置成False。
    '''
    def __init__(self, form_logits=True):
        self.__form_logists = form_logits
        self.__grad = None

    '''
    输入形状为(m, 1)
    '''
    def __call__(self, y_true, y_pred):
        if not self.__form_logists:
            loss = -y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)
            self.__grad = y_pred - y_true
            return loss

        #对数据进行自动缩小
        y_pred, diff = utils.reduce(y_pred)

        #转换成概率
        y_pred = dlmath.sigmoid(y_pred)
        #计算误差
        loss = -y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)

        #计算梯度
        grad = y_pred - y_true
        self.__grad = grad/scope

        return loss

    @property
    def gradient(self):
        return self.__grad

'''
多分类交叉熵损失函数
'''
class CategoricalCrossentropy(Loss):

    '''
    form_logits: 是否把输入数据转换成概率形式. 默认True. 如果输入数据已经是概率形式可以把这个参数设置成False。
    '''
    def __init__(self, form_logits=True):
        self.__form_logists = form_logits
        self.__grad = None

    '''
    输入形状为(m, n)
    '''
    def __call__(self, y_true, y_pred):
        if not self.__form_logists:
            loss = -y_true*np.log(y_pred)
            self.__grad = y_pred - y_true
            return loss

        #自动缩小
        y_pred, diff = utils.reduce(y_pred)

        #转换成概率分布
        y_pred = dlmath.prob_distribution(y_pred)
        #计算误差
        loss = -y_true*np.log(y_pred)
        #计算梯度
        grad = y_pred - y_true
        self.__grad = grad/diff

        return loss

    @property
    def gradient(self):
        return self.__grad
