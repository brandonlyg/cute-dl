import sys
sys.path.append("..")
sys.path.append("../cutedl")

import unittest
from unittest import TestCase
from cutedl import rnn_layers as rnn
import numpy as np

def display_params(params):
    print("params: ")
    for i in range(len(params)):
        p = params[i]
        print("name:", p.name, " value:", p.value, " gradient:", p.gradient)


'''
嵌入层测试
'''
class TestEmbedding(TestCase):
    @classmethod
    def setUpClass(cls):
        emb = rnn.Embedding(5, 10)
        emb.init_params()

        cls.emb = emb


    def testVecs(self):
        print("testVecs-----")
        emb = rnn.Embedding(5, 10, need_train=False)
        emb.init_params()
        vecs = emb.vectors

        in_batch = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 1]
        ]
        in_batch = np.array(in_batch)
        res = emb.forward(in_batch, False)
        print("res: ", res)

        tmp = np.zeros(in_batch.shape + (5,))
        m,_,_ = tmp.shape
        for i in range(m):
            tmp[i] = vecs[in_batch[i]]

        self.assertEqual(res.shape, tmp.shape)
        self.assertEqual(res.tolist(), tmp.tolist())

    def testTrain(self):
        print("test Train---------")
        emb = self.emb
        in_batch = [
            [0, 2, 3, 9, 2]
        ]
        in_batch = np.array(in_batch)
        out = emb.forward(in_batch, True)
        print("out: ", out)

        m, t, n = out.shape
        grad = np.arange(m*t*n).reshape(out.shape)
        emb.backward(grad)

        params = emb.params
        display_params(params)


'''
门控单元测试
'''
class TestGateUnit(TestCase):

    @classmethod
    def setUpClass(cls):
        print("start TestGateUnit-------")
        cls.gu = rnn.GateUnit(3, 2, None, 1)
        cls.gu.init_params()

    def testParams(self):
        print("test params ---")
        gu = self.gu
        display_params(gu.params)

    def test_train(self):
        print("test train ----")
        gu = self.gu

        in_batch = np.arange(4*2).reshape((4,2))
        hs = np.arange(4*3).reshape((4,3))
        out = gu.forward(in_batch, hs, True)

        self.assertEqual(out.shape, (4,3))
        print("out: ", out)

        m,n = out.shape
        grad = np.arange(m*n).reshape(out.shape)
        grad_in_batch, grad_hs = gu.backward(grad)
        self.assertEqual(grad_in_batch.shape, in_batch.shape)
        self.assertEqual(grad_hs.shape, hs.shape)
        print("grad_in_batch: ", grad_in_batch)
        print("grad_hs: ", grad_hs)

        display_params(gu.params)

'''
GRU 层测试
'''
class TestGRU(TestCase):

    @classmethod
    def setUpClass(cls):
        print("start test GRU ----------------")
        cls.gru = rnn.GRU(4, 3)
        cls.gru.init_params()

    def test_params(self):
        print("test params --")
        gru = self.gru
        display_params(gru.params)

    def test_train(self):
        print("test train")
        gru = self.gru

        in_batch = np.random.uniform(0, 1, (2,2,3)) * 0.1
        out = gru.forward(in_batch, True)
        self.assertEqual(out.shape, (2,2,4))
        print("out: ", out)

        grad = np.random.uniform(0, 1, (2,2,4))*0.1
        grad_in_batch = gru.backward(grad)
        self.assertEqual(grad_in_batch.shape, in_batch.shape)
        print("grad_in_batch: ", grad_in_batch)

        display_params(gru.params)

'''
测试LSTM
'''
class TestLSTM(TestCase):

    @classmethod
    def setUpClass(cls):
        print("start test LSTM ----------------")
        cls.lstm = rnn.LSTM(4, 3)
        cls.lstm.init_params()

    def test_params(self):
        print("test params")

        lstm = self.lstm
        display_params(lstm.params)

    def test_train(self):
        print("test train")
        lstm = self.lstm

        in_batch = np.random.uniform(0, 1, (2,2,3)) * 0.1
        out = lstm.forward(in_batch, True)
        self.assertEqual(out.shape, (2,2,4))
        print("out: ", out)

        grad = np.random.uniform(0, 1, (2,2,4))*0.1
        grad_in_batch = lstm.backward(grad)
        self.assertEqual(grad_in_batch.shape, in_batch.shape)
        print("grad_in_batch: ", grad_in_batch)

        display_params(lstm.params)



if __name__ == '__main__':
    unittest.main()
