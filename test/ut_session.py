# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl.model import Model
from cutedl.session import Session
from cutedl import losses, optimizers
import numpy as np
from example_layer import Simplelayer

'''
session单元测试
'''
class TestSession(TestCase):

    @classmethod
    def setUpClass(cls):
        model = Model([
                Simplelayer(4, inshape=3),
                Simplelayer(5),
                Simplelayer(1)
            ])
        model.assemble()

        cls.sess = Session(
                    model,
                    loss=losses.Mse(),
                    optimizer=optimizers.Fixed()
                )

    def test_batch_train(self):
        in_batch = np.random.randn(1000, 3)
        y_true = np.ones((1000, 1))
        loss = self.sess.batch_train(in_batch, y_true)
        print("loss:", loss)

    def test_save_load(self):
        fpath = "models/test_m1"
        self.sess.save(fpath)

        sess = Session.load(fpath)


if __name__ == '__main__':
    unittest.main()
