# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl import cnn_layers as cnn
from cutedl import nn_layers as nn
from cutedl.model import Model
import numpy as np


'''
卷积层单元测试
'''

'''
2D卷积层测试
'''
class TestConv2D(TestCase):

    @classmethod
    def setUpClass(cls):
        model = Model([
                cnn.Conv2D(3, (3,3), inshape=(2,12,12)),
                cnn.Conv2D(4, (5, 5), padding='valid'),
                nn.Dense(5)
                ])
        model.assemble()

        cls.model = model

    '''
    测试输入输出形状, 参数形状
    '''
    def test_shape(self):
        print("test shape")
        model = self.model

        conv0 = model.get_layer(0)
        conv1 = model.get_layer(1)
        dense = model.get_layer(2)

        print("conv0 inshape: %s, outshape: %s"%(str(conv0.inshape), str(conv0.outshape)))
        self.assertEqual(conv0.inshape, (2,12,12))
        self.assertEqual(conv0.outshape, (3,12,12))
        W = conv0.params[0]
        b = conv0.params[1]
        print("conv0 params shape. W: ", W.value, " b: ", b.value)
        self.assertEqual(W.value.shape, (2*3*3, 3))
        self.assertEqual(b.value.shape, (3,))

        print("conv1 inshape: %s, outshape: %s"%(str(conv1.inshape), str(conv1.outshape)))
        self.assertEqual(conv1.inshape, (3,12,12))
        self.assertEqual(conv1.outshape, (4*8*8,))
        W = conv1.params[0]
        b = conv1.params[1]
        print("conv1 params shape. W: ", W.value, " b: ", b.value)
        self.assertEqual(W.value.shape, (3*5*5, 4))
        self.assertEqual(b.value.shape, (4,))

        print("dense inshape: %s, outshape: %s"%(str(dense.inshape), str(dense.outshape)))
        self.assertEqual(dense.inshape, (4*8*8, ))
        self.assertEqual(dense.outshape, (5,))
        W = dense.params[0]
        b = dense.params[1]
        print("conv1 params shape. W: ", W.value, " b: ", b.value)
        self.assertEqual(W.value.shape, (4*8*8, 5))
        self.assertEqual(b.value.shape, (5,))

    '''
    向前传播，反向传播
    '''
    def test_forward_backward(self):
        print("test forward and backward")
        model = self.model

        in_batch = np.random.uniform(size=(2, 2, 12, 12))
        out_batch = model.predict(in_batch, training=True)

        model.backward(out_batch)


if '__main__' == __name__:
    unittest.main()
