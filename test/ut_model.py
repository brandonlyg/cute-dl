# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import pdb

import unittest
from unittest import TestCase
from cutedl.model import Layer, LayerParam, Model
import numpy as np
from example_layer import Simplelayer

'''
模型框架测试
'''
class TestLayer(TestCase):

    '''
    def setUp(self):
        self.ly_0 = Simplelayer((3,3), inshape=2)
        self.ly_1 = Simplelayer((4,4))

        self.ly_1.join(self.ly_0)
    '''

    @classmethod
    def setUpClass(cls):
        cls.ly_0 = Simplelayer((3,3), inshape=2)
        cls.ly_1 = Simplelayer((4,4))

        cls.ly_1.join(cls.ly_0)

        cls.ly_0_p = np.arange(2*3*3).reshape((2,3,3))
        cls.ly_1_p = np.arange(3*3*4*4).reshape((3,3,4,4))

    #检查输入输出的形状
    def test_shape(self):
        print("test layer shape")
        self.assertEqual(self.ly_0.inshape, (2,))
        self.assertEqual(self.ly_0.outshape, (3,3))

        self.assertEqual(self.ly_1.inshape, (3,3))
        self.assertEqual(self.ly_1.outshape, (4,4))

    #检查参数
    def test_params(self):
        print("test layer params")
        params = self.ly_0.params
        self.assertTrue(len(params) == 1)
        p = params[0]
        self.assertEqual(p.value.tolist(), self.ly_0_p.tolist())

        params = self.ly_1.params
        self.assertTrue(len(params)==1)
        p = params[0]
        self.assertEqual(p.value.tolist(), self.ly_1_p.tolist())

    #更新参数
    def test_update_param(self):
        print("test layer update params")
        p = self.ly_0.params[0]
        p_val = np.ones(p.value.shape)
        p_grad = np.ones(p.value.shape) + 1

        p.value = p_val
        p.gradient = p_grad

        #pdb.set_trace()

        p1 = self.ly_0.params[0]
        self.assertEqual(p1.value.tolist(), p_val.tolist())
        self.assertEqual(p1.gradient.tolist(), p_grad.tolist())

    #测试正反向传播
    def test_forward_backward(self):
        print("test layer forward")
        ly = Simplelayer(4, inshape=3)
        in_batch = np.arange(6).reshape((2,3))
        p = np.arange(12).reshape(3,4)
        out_batch = ly.forward(in_batch)
        self.assertEqual(out_batch.tolist(), (in_batch @ p).tolist())

        print("test layer backward")
        gradient = np.arange(8).reshape((2,4))
        out_grad = ly.backward(gradient)
        self.assertEqual(out_grad.tolist(), (gradient @ p.T).tolist())


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = Model([
                    Simplelayer(3, inshape=2),
                    Simplelayer(4),
                    Simplelayer(1)
                ])

        cls.model.assemble()

    def test_basal(self):
        print("basel test model")

        print("check model layer shape")
        shapes = [((2,), (3,)), ((3,), (4,)), ((4,),(1,))]
        idx = 0
        for ly in self.model.layer_iterator():
            self.assertEqual(ly.inshape, shapes[idx][0])
            self.assertEqual(ly.outshape, shapes[idx][1])
            idx += 1

        print("check model layer count")
        self.assertTrue(idx==3)

        print("check model layer param shape")
        shapes = [(2,3), (3,4), (4,1)]
        for i in range(idx):
            ly = self.model.get_layer(i)
            param = ly.params[0]
            self.assertEqual(param.value.shape, shapes[i])

    def test_update_param(self):
        print("test model update param")

        for ly in self.model.layer_iterator():
            ly_0 = ly
            break

        p = ly_0.params[0]
        print("before update. param value:", p.value)
        val = np.ones(p.value.shape)
        p.value = val

        ly_0 = None
        for ly in self.model.layer_iterator():
            ly_0 = ly
            break

        print("after update. param value:", ly_0.params[0].value)

        self.assertTrue(ly_0.params[0].value.tolist() == val.tolist())

    def test_predict(self):
        print("test model predict")
        in_batch = np.random.randn(1000, 2)
        y_pred = self.model.predict(in_batch)
        self.assertEqual(y_pred.shape, (1000, 1))

    def test_backward(self):
        print("test model backward")
        gradient = np.random.randn(1000, 1)
        self.model.backward(gradient)

    def test_save_load(self):
        print("test model save and load")
        fpath = "./models/testmodel"
        self.model.save(fpath)

        model = Model.load(fpath)


if __name__ == '__main__':
    unittest.main()
