# coding=utf-8

import numpy as np

'''
实现一些常用的数学函数
'''

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

'''
转换成分布列
x shape (m, ..., n)
'''
def prob_distribution(x):
    xshape = x.shape
    x = x.reshape((-1, xshape[-1]))

    expval = np.exp(x)
    sum = expval.sum(axis=1).reshape(-1,1)

    prob_d = expval/sum
    prob_d = prob_d.reshape(xshape)

    return prob_d


'''
类别抽样
categories 类别确信度 shape=(m, ..., n)
count 抽样数量
'''
def categories_sample(categories, count):
    #转换成分布列
    prob_d= prob_distribution(categories)
    shape = prob_d.shape
    #转换成分布
    prob_d = prob_d.reshape((-1, shape[-1]))
    m, n = prob_d.shape
    for i in range(1, n):
        prob_d[:, i] += prob_d[:, i-1]

    prob_d[:, n-1] = 1.0

    #随机抽样
    res = np.zeros((m, count))
    for i in range(count):
        p = np.random.uniform(0, 1, (m,)).reshape((m, 1))
        item = (prob_d < p).astype(int)
        item = item.sum(axis=1)
        res[:, i] = item

    res = res.reshape(shape[0:len(shape)-1]+(count,)).astype(int)
    return res
