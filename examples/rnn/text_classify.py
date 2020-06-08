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
from cutedl import wrapper_layers as wrapper

report_path = "./pics/imdbr-recoginze-"
model_path = "./model/imdbr-recoginze-"

def fit(name, model):
    sess = Session(model,
                loss = losses.BinaryCrossentropy(),
                optimizer = optimizers.Adam()
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 10)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 100, val_data=ds_test, val_batches=20,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=fit_tools.binary_accuracy),
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


def fit_gru():
    print("fit gru")
    model = Model([
                rnn.Embedding(64, vocab_size+1),
                wrapper.Bidirectional(rnn.GRU(64), rnn.GRU(64)),
                nn.Filter(),
                nn.Dense(64),
                nn.Dense(1, activation='linear')
            ])
    model.assemble()
    fit('gru', model)

def fit_lstm():
    print("fit lstm")
    model = Model([
                rnn.Embedding(64, vocab_size+1),
                wrapper.Bidirectional(rnn.LSTM(64), rnn.LSTM(64)),
                nn.Filter(),
                nn.Dense(64),
                nn.Dense(1, activation='linear')
            ])
    model.assemble()
    fit('lstm', model)


if '__main__' == __name__:
    fit_gru()
    #fit_lstm()
    pass
