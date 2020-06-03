# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl import nn_layers as nn
import numpy as np

'''
通用神经网络层单元测试
'''

class TestNNLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ly_dense = nn.Dense(4, inshape=2, activation='relu')
        cls.ly_dropout = nn.Dropout(keep_prob=0.5)
        cls.ly_dense1 = nn.Dense(6)

        cls.ly_dropout.set_prev(cls.ly_dense)
        cls.ly_dense1.set_prev(cls.ly_dropout)

        cls.ly_dense.init_params()
        cls.ly_dropout.init_params()
        cls.ly_dense1.init_params()

    '''
    测试Dense层
    '''
    def testDense(self):
        print("test dense")

        inb = np.arange(4).reshape((2,2))
        out = self.ly_dense.forward(inb)
        self.assertEqual(out.shape, (2,4))

        gradient = np.arange(8).reshape((2,4))
        out = self.ly_dense.backward(gradient)
        self.assertEqual(out.shape, (2,2))

    '''
    测试dropout层
    '''
    def testDropout(self):
        print("test dropout")

        self.assertEqual(self.ly_dropout.inshape, self.ly_dropout.outshape)
        self.assertEqual(self.ly_dropout.outshape, self.ly_dense.outshape)
        self.assertEqual(self.ly_dense1.inshape, self.ly_dense.outshape)
        self.assertEqual(self.ly_dense1.outshape, (6,))

        inb = np.arange(2, 2+8).reshape((2,4))
        out = self.ly_dropout.forward(inb)
        print("training=False dropout forward: ", out)

        bout = self.ly_dropout.backward(inb)
        print("training=False dropout backward: ", bout)

        self.assertEqual(out.tolist(), bout.tolist())

        inb = np.arange(2, 2+8).reshape((2,4))
        out = self.ly_dropout.forward(inb, training=True)
        print("training=True dropout forward: ", out)

        bout = self.ly_dropout.backward(inb)
        print("training=True dropout backward: ", bout)

        self.assertEqual(out.tolist(), bout.tolist())


class TestFilter(TestCase):

    def test_all(self):
        print("test Filter all")

        filter = nn.Filter()
        in_batch = np.arange(3*3*4).reshape(3,3,4)
        out = filter.forward(in_batch)
        print("out: ", out)
        self.assertEqual(out.tolist(), in_batch[:,-1,:].tolist())

        grad = np.arange(3 * 4).reshape(3,4)
        grad_out = filter.backward(grad)
        print("grad_out: ", grad_out)
        self.assertEqual(grad_out.shape, in_batch.shape)


if __name__ == '__main__':
    unittest.main()
