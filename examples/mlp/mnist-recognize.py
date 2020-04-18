# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
手写数字识别模型
'''

from datasets import mnist

'''
加载手写数字数据集
'''
ds_train, ts_test = mnist.load()

from cutedl.model import Model
from cutedl.session import Session, FitListener
from cutedl import nn_layers as nn
from cutedl import losses, optimizers

'''
训练模型
'''
def fit():
    inshape = ds_train.data.shape[1]
    model = Model([
                nn.Dense(10, inshape=inshape, activation='relu')
            ])
    model.assemble()

    sess = Session(
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.Fixed()
            )
