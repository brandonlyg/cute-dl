# coding=utf-8

'''
优化器单元测试
'''

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import pdb

import unittest
from unittest import TestCase
from cutedl.model import Layer, LayerParam, Model
from cutedl.optimizers import Momentum, Adagrad, RMSProp, Adadelta, Adam
import numpy as np
from example_layer import Simplelayer


class TestLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = Model([
                        Simplelayer(3, inshape=2),
                        Simplelayer(4)
                    ])
        cls.model.assemble()



    def test_momentum(self):
        print("test Momentum")
        mom = Momentum()
        model = self.model

        x_in = np.arange(3*2).reshape(3,2)
        y_ = model.predict(x_in, training=True)

        grad = np.ones(y_.shape)
        model.backward(grad)

        mom(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " momentum:", p.momentum)

        mom(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " momentum:", p.momentum)

    def test_adagrad(self):
        print("test Adagrad")
        adagrad = Adagrad()
        model = self.model

        x_in = np.arange(3*2).reshape(3,2)
        y_ = model.predict(x_in, training=True)

        grad = np.ones(y_.shape)
        model.backward(grad)

        adagrad(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adagrad:", p.adagrad)

        adagrad(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adagrad:", p.adagrad)

    def test_rmsprop(self):
        print("test RMSProp")
        rmsprop = RMSProp()
        model = self.model

        x_in = np.arange(3*2).reshape(3,2)
        y_ = model.predict(x_in, training=True)

        grad = np.ones(y_.shape)
        model.backward(grad)

        rmsprop(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " rmsprop_storeup:", p.rmsprop_storeup)

        rmsprop(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                #pdb.set_trace()
                print("name:", p.name, " rmsprop_momentum:", p.rmsprop_storeup)

    def test_adadelta(self):
        print("test Adadelta")
        adadelta = Adadelta()
        model = self.model

        x_in = np.arange(3*2).reshape(3,2)
        y_ = model.predict(x_in, training=True)

        grad = np.ones(y_.shape)
        model.backward(grad)

        adadelta(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adadelta_storeup:", p.adadelta_storeup)

        adadelta(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adadelta_predelta:", p.adadelta_predelta)

    def test_adam(self):
        print("test Adm")
        adam = Adam()
        model = self.model

        x_in = np.arange(3*2).reshape(3,2)
        y_ = model.predict(x_in, training=True)

        grad = np.ones(y_.shape)
        model.backward(grad)

        adam(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adam_momentum:", p.adam_momentum)
                print("name:", p.name, " adam_mdpr_t:", p.adam_mdpr_t)
                print("name:", p.name, " adam_storeup:", p.adam_storeup)
                print("name:", p.name, " adam_sdpr_t:", p.adam_sdpr_t)

        adam(model)
        for ly in model.layer_iterator():
            for p in ly.params:
                print("name:", p.name, " adam_momentum:", p.adam_momentum)
                print("name:", p.name, " adam_mdpr_t:", p.adam_mdpr_t)
                print("name:", p.name, " adam_storeup:", p.adam_storeup)
                print("name:", p.name, " adam_sdpr_t:", p.adam_sdpr_t)


if __name__ == '__main__':
    unittest.main()
