# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl import activations as act
import numpy as np

'''
激活函数单元测试
'''

class TestRelu(TestCase):

    #在运行测试用例之前执行
    def setUp(self):
        self.relu = act.get('relu')

    def test_call(self):
        inp = np.array([[-1, 1], [0, 0.1]])
        outp = self.relu(inp)

        self.assertEqual(outp.tolist(), [[0,1],[0,0.1]])

    def test_grad(self):
        inp = np.array([[-1, 1], [0, 0.1]])
        self.relu(inp)

        inp = np.array([[1,2], [3,4]])
        outp = self.relu.grad(inp)

        self.assertEqual(outp.tolist(), [[0,2], [0,4]])


if __name__ == '__main__':
    unittest.main()
