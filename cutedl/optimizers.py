# coding=utf-8

import pdb
import re

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
            self.update_param(model, p)

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
    def update_param(self, model, param):
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

    def update_param(self, model, param):
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

    def update_param(self, model, param):
        #pdb.set_trace()
        param.value = (1 - self.__damping) * param.value


'''
固定学习率优化器
'''
class Fixed(LROptimizer):

    '''
    lt: 学习率
    '''
    def __init__(self, lt=0.01):
        self.__lt = lt

    def update_param(self, model, param):
        #pdb.set_trace()
        param.value -= self.__lt * param.gradient
        param.udt += 1
