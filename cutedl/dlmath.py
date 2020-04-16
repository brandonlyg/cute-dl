# coding=utf-8

import numpy as np

'''
实现一些常用的数学函数
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))

def prob_distribution(x):
    expval = np.exp(x)
    sum = expval.sum()

    prob_d = expval/(sum + 1e-8)

    return prob_d
