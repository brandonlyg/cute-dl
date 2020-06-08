# coding=utf-8

'''
测试包装层
'''

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import pdb
import unittest
from unittest import TestCase
from cutedl import wrapper_layers as wrapper
from cutedl import rnn_layers as rnn
import numpy as np

import test_utils as tu


'''
双向层
'''
class TestBidirectional(TestCase):

    @classmethod
    def setUpClass(cls):
        print("start test Bidirectional ------------------")

        bid = wrapper.Bidirectional(rnn.GRU(4, 2), rnn.GRU(4, 2))
        #pdb.set_trace()
        bid.set_parent(None)
        bid.init_params()

        cls.bid = bid

    def test_params(self):
        print("test params ---")
        bid = self.bid

        tu.display_params(bid.params)
        print("inshape: ", bid.inshape, " outshape: ", bid.outshape)

    def test_train(self):
        print("test train --")
        #pdb.set_trace()
        bid = self.bid
        in_batch = np.random.uniform(-1, 1, (3, 3, 2))
        out = bid.forward(in_batch, training=True)
        self.assertEqual(out.shape, (3, 3, 8))
        print("out: ", out)

        grad = np.random.uniform(-1, 1, (3, 3, 8))
        grad_out = bid.backward(grad)
        self.assertEqual(grad_out.shape, (3,3,2))
        print("grad_out: ", grad_out)


if __name__ == '__main__':
    unittest.main()
