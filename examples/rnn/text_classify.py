# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../datasets')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
在imdbr数据集上执行分类任务
'''

import pdb
import numpy as np

from datasets.mnist import Mnist
from datasets.imdbr import IMDBR
from datasets import fit_tools

#加载数据集
imdbr = IMDBR("../datasets/imdbr")
ds_train, ds_test, vocab_size = imdbr.load_ds(batch_size=64)

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import rnn_layers as rnn

report_path = "./pics/mnist-recoginze-"
model_path = "./model/mnist-recoginze-"
def fit():
    model = Model([
                rnn.Embedding(32, vocab_size+1),
                rnn.GRU(64),
                nn.Filter(),
                nn.Dense(64),
                nn.Dense(1, activation='linear')
            ])
    model.assemble()

    sess = Session(model,
                loss = losses.BinaryCrossentropy(),
                optimizer = optimizers.Adam()
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 20)

    accuracy = lambda h:fit_tools.binary_accuracy(ds_test.label.reshape((-1, 1,)), h)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 3, val_data=ds_test, 
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=accuracy),
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


if '__main__' == __name__:
    fit()