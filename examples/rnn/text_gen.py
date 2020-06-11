# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../datasets')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
古诗生成模型
'''

import pdb
import numpy as np

from datasets.china_poetry import ChinaPoetry
from datasets import fit_tools

ds = ChinaPoetry("../datasets/china_poetry")
ds_train = ds.load_ds(64, 7)
vocab = ds.vocabulary

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import rnn_layers as rnn
from cutedl import wrapper_layers as wrapper

report_path = "./pics/china-poetry-"
model_path = "./model/china-poetry-"

def fit(name, model):
    sess = Session(model,
                loss = losses.SparseCategoricalCrossentropy(),
                optimizer = optimizers.Adam()
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'loss', 10)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 100,
                        listeners=[
                            stop_fit,
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


def fit_gru():
    vocab_size = vocab.size()
    model = Model([
                rnn.Embedding(128, vocab_size+1, batch_size=64),
                rnn.GRU(128),
                nn.Dense(vocab_size, activation='linear')
            ])

    model.assemble()
    fit("gru", model)


def gen_text():
    mpath = model_path+"gru"

    model = Model.load(mpath)

    


if '__main__' == __name__:
    fit_gru()
