# coding=utf-8

import pdb

'''
学习率优化器
'''
class Optimizer(object):

    '''
    更新参数
    '''
    def __call__(self, model):
        raise Exception('not implement')


'''
固定学习率优化器
'''
class Fixed(Optimizer):

    '''
    lt: 学习率
    '''
    def __init__(self, lt=0.01):
        self.__lt = lt

    def __call__(self, model):
        #pdb.set_trace()
        for ly in model.layer_iterator():
            for p in ly.params:
                p.value -= self.__lt * p.gradient
                p.udt += 1
