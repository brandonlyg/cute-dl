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
x shape (m, n)
'''
def prob_distribution(x):
    expval = np.exp(x)
    sum = expval.sum(axis=1).reshape(-1,1) + 1e-8

    prob_d = expval/sum

    return prob_d


'''
类别抽样
categories 类别确信度 shape=(m, n)
count 抽样数量
'''
def categories_sample(categories, count):
    #转换成分布列
    prob_d= prob_distribution(categories)

    #转换成分布
    m, n = prob_d.shape
    p_sum = prob_d[:, 0]
    for i in range(1, n):
        prob_d_col[: i] += p_sum
        p_sum = prob_d_col[:, i]

    #随机抽样
    res = np.zeros((count, m))
    for i in range(count):
        p = np.uniform(0, 1, m)
        item = (prob_d < p).astype(int)
        item = item.sum(axis=1)
        res[i] = item

    return res
