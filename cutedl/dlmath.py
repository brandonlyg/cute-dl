# coding=utf-8

import numpy as np

'''
实现一些常用的数学函数
'''

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def prob_distribution(x):
    expval = np.exp(x)
    sum = expval.sum(axis=1) + 1e-8

    prob_d = expval/sum

    return prob_d
