# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl import losses
import numpy as np

'''
损失函数单元测试
'''

class TestLose(TestCase):

    def test_mse(self):
        print("test Mse")
        mse = losses.Mse()

        y_true = np.array([[3], [5], [7]])
        y_pred = np.array([[1], [2], [3]])

        loss = mse(y_true, y_pred)
        print("loss: ", loss)

    '''
    测试二元交叉熵损失函数
    '''
    def test_binary_cross_entropy(self):
        print("test BinaryCrossentropy")
        loss = losses.BinaryCrossentropy()

        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[3], [5], [2], [6]])

        lres = loss(y_true, y_pred)
        print("loss: ", lres)
        self.assertEqual(loss.gradient.shape, (4, 1))

        y_pred = np.array([[-0.5], [0.7], [-0.2], [0.3]])
        lres = loss(y_true, y_pred)
        print("loss: ", lres)

    '''
    多类别交叉熵损失函数
    '''
    def test_categorical_cross_entropy(self):
        print("test CategoricalCrossentropy")
        cce = losses.CategoricalCrossentropy()

        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[3, 2, 1], [6, 4, 8], [5, 3, 1]])

        loss = cce(y_true, y_pred)
        print("loss: ", loss)
        self.assertEqual(cce.gradient.shape, (3, 3))

        y_pred = np.array([[-0.5, 0.2, 0.1], [-0.1, 0.1, 0.05], [-0.1, 0.2, -0.2]])
        loss = cce(y_true, y_pred)
        print("loss: ", loss)

    def test_spare_categorical_cross_entropy(self):
        print("test SparseCategoricalCrossentropy")
        cce = losses.SparseCategoricalCrossentropy()

        y_true = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        y_pred = np.random.uniform(-1, 1, (3, 3, 3))



if __name__ == '__main__':
    unittest.main()
