# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl.model import Model
from cutedl.session import Session, FitListener
from cutedl.dataset import Dataset
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

        l2 = optimizers.L2(0.5)
        cls.sess = Session(
                    model,
                    loss=losses.Mse(),
                    optimizer=optimizers.Fixed(),
                    genoptms=[l2]
                )

    def test_batch_train(self):
        in_batch = np.random.randn(1000, 3)
        y_true = np.ones((1000, 1))
        loss = self.sess.batch_train(in_batch, y_true)
        print("loss:", loss)

    def test_fit(self):
        d_x = np.arange(160 * 3).reshape((160,3))
        d_y = np.arange(160).reshape((160, 1))
        ds = Dataset(d_x, d_y, 64)
        self.assertEqual(ds.batch_count, 3)

        val_x = np.arange(64 * 3).reshape((64,3))
        val_y = np.arange(64).reshape((64, 1))
        val_ds = Dataset(val_x, val_y, 64)

        def on_epoch_start(history):
            print("on_epoch_start")

        def on_epoch_end(history):
            print("on_epoch_end")

        def on_val_start(history):
            print("on_val_start")

        def on_val_end(history):
            print("on_val_end")
            if len(history['loss']) > 7:
                self.sess.stop_fit()

        history = self.sess.fit(ds, 10, val_data=val_ds,
                                listeners=[
                                    FitListener('epoch_start', callback=on_epoch_start),
                                    FitListener('epoch_end', callback=on_epoch_end),
                                    FitListener('val_start', callback=on_val_start),
                                    FitListener('val_end', callback=on_val_end)
                                    ]
                                )
        print("history: ", history)


    def test_save_load(self):
        fpath = "models/test_m1"
        self.sess.save(fpath)

        sess = Session.load(fpath)


if __name__ == '__main__':
    unittest.main()
