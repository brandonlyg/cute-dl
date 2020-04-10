# coding=utf-8

import sys
sys.path.append("..")

import unittest
from unittest import TestCase
from cutedl import losses
import numpy as np

'''
损失函数单元测试
'''

class TestLose(TestCase):

    def test_mse(self):
        mse = loss.Mse()

        y_true = np.array([[3], [5], [7]])
        y_pred = np.array([[1], [2], [3]])

        lres = mse(y_true, y_pred)

        self.assertTrue(lres - 29/2 < 0.00001)

        tmp = np.array([[2],[3],[4]])/3
        self.assertEqual(mse.gradient.tolist(), tmp.tolist())


if __name__ == '__main__':
    unittest.main()
