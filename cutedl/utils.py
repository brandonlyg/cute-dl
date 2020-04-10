# coding=utf-8

import numpy as np

'''
一些工具函数
'''

'''
把多维形状展平成一维
'''
def flat_shape(*shape):
    #print("shape:", shape)
    if len(shape) == 0:
        return 1

    if len(shape) == 1:
        shape = shape[0]

    if type(shape) != type(()):
        return shape

    res = 1
    for item in shape:
        if type(item) == type(()):
            if len(item) == 1:
                res *= item[0]
            elif len(item) > 1:
                res = res * flat_shape(item)
        else:
            res *= item

    return res

if '__main__' == __name__:
    print("flat_shape (2,):", flat_shape((2,)))
    print("flat_shape (2,3):", flat_shape((2,3)))
    print("flat_shape (2, (3,), (4,5))", flat_shape((2, (3,), (4,5))))
