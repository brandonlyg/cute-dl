# coding=utf-8

import pdb
import re
import numpy as np

'''
所有优化器的父类
'''
class Optimizer(object):

    '''
    更新参数
    '''
    def __call__(self, model):
        params = self.match(model)
        for p in params:
            self.update_param(p)
            p.udt += 1

    '''
    pattern: 正则表达式匹配模式
    '''
    @property
    def pattern(self):
        return '^/.+/weight.*'

    '''
    得到名字匹配pattern的参数
    '''
    def match(self, model):
        params = []
        rep = re.compile(self.pattern)
        for ly in model.layer_iterator():
            for p in ly.params:
                if rep.match(p.name) is None:
                    continue

                params.append(p)

        return params

    '''
    更新指定的参数。由子类实现。
    model Model对象
    param LayerParam对象
    '''
    def update_param(self, param):
        raise Exception("not impliment")

'''
所有学习率优化器的基础类
'''
class LROptimizer(Optimizer):

    @property
    def pattern(self):
        return '^/.+/(weight|bias).*'

'''
L1 正则化
'''
class L1(Optimizer):
    '''
    damping 参数衰减率
    '''
    def __init__(self, damping):
        self.__damping = damping

    def update_param(self, param):
        norm = math.sqrt((param.value**2).sum()) + 1e-8
        param.value = (1 - self.__damping/norm) * param.value


'''
L2 正则化
'''
class L2(Optimizer):
    '''
    damping 参数衰减率
    '''
    def __init__(self, damping):
        self.__damping = damping

    def update_param(self, param):
        #pdb.set_trace()
        param.value = (1 - self.__damping) * param.value


'''
固定学习率优化器
'''
class Fixed(LROptimizer):

    '''
    lr: 学习率
    '''
    def __init__(self, lr=0.01):
        self.__lr = lr

    def update_param(self, param):
        #pdb.set_trace()
        param.value -= self.__lr * param.gradient

'''
学习率优化器: 动量算法
'''
class Momentum(LROptimizer):
    '''
    lr: 学习率
    dpr: 动量衰减率0<dpr<1
    '''
    def __init__(self, lr=1e-4, dpr=0.9):
        self.__lr = lr
        self.__dpr = dpr
        if dpr <= 0 or dpr >= 1:
            raise Exception("invalid dpr:%f"%dpr)

    def update_param(self, param):
        #pdb.set_trace()
        if not hasattr(param, 'momentum'):
            #为参数添加动量属性
            param.momentum = np.zeros(param.value.shape)

        param.momentum = param.momentum * self.__dpr + param.gradient * self.__lr

        param.value -= param.momentum


'''
学习率优化器: Adagrad算法
'''
class Adagrad(LROptimizer):

    def __init__(self, lr=0.1):
        self.__lr = lr

    def update_param(self, param):
        #pdb.set_trace()
        if  not hasattr(param, 'adagrad'):
            #添加积累量属性
            param.adagrad = np.zeros(param.value.shape)

        a = 1e-6
        param.adagrad += param.gradient ** 2
        grad = self.__lr/np.sqrt(param.adagrad + a) * param.gradient
        param.value -= grad


'''
学习率优化器: RMSProp算法
'''
class RMSProp(LROptimizer):

    '''
    sdpr: 积累量衰减率 0<sdpr<1
    '''
    def __init__(self, lr=1e-4, sdpr=0.99):
        self.__lr = lr
        self.__sdpr = sdpr

        if sdpr <= 0 or sdpr >= 1:
            raise Exception("invalid sdpr:%f"%sdpr)

    def update_param(self, param):
        #pdb.set_trace()
        if not hasattr(param, 'rmsprop_storeup'):
            #添加积累量属性
            param.rmsprop_storeup = np.zeros(param.value.shape)

        a = 1e-6

        param.rmsprop_storeup = param.rmsprop_storeup * self.__sdpr + (param.gradient**2) * (1-self.__sdpr)
        grad = self.__lr/np.sqrt(param.rmsprop_storeup + a) * param.gradient

        param.value -= grad

'''
学习率优化器: Adadelta算法
'''
class Adadelta(LROptimizer):

    '''
    dpr: 积累量衰减率 0<dpr<1
    '''
    def __init__(self, dpr=0.99):
        self.__dpr = dpr

        if dpr <= 0 or dpr >= 1:
            raise Exception("invalid dpr:%f"%dpr)

    def update_param(self, param):
        #pdb.set_trace()
        if not hasattr(param, 'adadelta_storeup'):
            #添加积累量属性
            param.adadelta_storeup = np.zeros(param.value.shape)

        if not hasattr(param, "adadelta_predelta"):
            #添加上步的变化量属性
            param.adadelta_predelta = np.zeros(param.value.shape)

        a = 1e-6

        param.adadelta_storeup = param.adadelta_storeup * self.__dpr + (param.gradient**2)*(1-self.__dpr)
        grad = (np.sqrt(param.adadelta_predelta) + a)/(np.sqrt(param.adadelta_storeup) + a) * param.gradient
        param.adadelta_predelta = param.adadelta_predelta * self.__dpr + (grad**2)*(1-self.__dpr)

        param.value -= grad


'''
学习率优化器: Adam算法
'''
class Adam(LROptimizer):

    '''
    mdpr: 动量衰减率 0<mdpr<1
    sdpr: 积累量衰减率 0<sdpr<1
    '''
    def __init__(self, lr=1e-4, mdpr=0.9, sdpr=0.99):
        self.__lr = lr
        self.__mdpr = mdpr
        self.__sdpr = sdpr

        if mdpr <= 0 or mdpr >= 1:
            raise Exception("invalid mdpr:%f"%mdpr)

        if sdpr <= 0 or sdpr >= 1:
            raise Exception("invalid sdpr:%f"%sdpr)

    def update_param(self, param):
        #pdb.set_trace()
        if not hasattr(param, 'adam_momentum'):
            #添加动量属性
            param.adam_momentum = np.zeros(param.value.shape)

        if not hasattr(param, 'adam_mdpr_t'):
            #mdpr的t次方
            param.adam_mdpr_t = 1

        if not hasattr(param, 'adam_storeup'):
            #添加积累量属性
            param.adam_storeup = np.zeros(param.value.shape)

        if not hasattr(param, 'adam_sdpr_t'):
            #动量sdpr的t次方
            param.adam_sdpr_t = 1

        a = 1e-8
        #计算动量
        param.adam_momentum = param.adam_momentum * self.__mdpr + param.gradient * (1-self.__mdpr)
        #偏差修正
        param.adam_mdpr_t *= self.__mdpr
        momentum = param.adam_momentum/(1-param.adam_mdpr_t)

        #计算积累量
        param.adam_storeup = param.adam_storeup * self.__sdpr + (param.gradient**2) * (1-self.__sdpr)
        #偏差修正
        param.adam_sdpr_t *= self.__sdpr
        storeup = param.adam_storeup/(1-param.adam_sdpr_t)

        grad = self.__lr * momentum/(np.sqrt(storeup)+a)
        param.value -= grad
