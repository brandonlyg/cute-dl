# coding=utf-8

import pdb

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

'''
多分类交叉熵损失函数
'''
