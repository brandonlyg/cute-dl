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
from cutedl import dlmath

report_path = "./pics/text-gen-"
model_path = "./model/text-gen-"

def fit(name, model):
    sess = Session(model,
                loss = losses.SparseCategoricalCrossentropy(),
                optimizer = optimizers.Adam()
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'loss', 100)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 1000,
                        listeners=[
                            stop_fit,
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


def fit_gru():
    vocab_size = vocab.size()
    print("vocab size: ", vocab_size)
    model = Model([
                rnn.Embedding(256, vocab_size),
                rnn.GRU(1024, stateful=True),
                nn.Dense(1024),
                nn.Dropout(0.5),
                nn.Dense(vocab_size, activation='linear')
            ])

    model.assemble()
    fit("gru", model)


def gen_text():
    mpath = model_path+"gru"

    model = Model.load(mpath)
    print("loadding model finished")
    outshape = (4, 7)

    print("vocab size: ", vocab.size())

    def do_gen(txt):
        #编码
        #pdb.set_trace()
        res = vocab.encode(sentence=txt)

        m, n = outshape

        for i in range(m*n - 1):
            in_batch = np.array(res).reshape((1, -1))
            preds = model.predict(in_batch)
            #取最后一维的预测结果
            preds = preds[:, -1]
            outs = dlmath.categories_sample(preds, 1)
            res.append(outs[0,0])

        #pdb.set_trace()
        txt = ""
        for i in range(m):
            txt = txt + ''.join(vocab.decode(res[i*n:(i+1)*n])) + "\n"

        return txt


    starts = ['云', '故', '画', '花']
    for txt in starts:
        model.reset()
        res = do_gen(txt)
        print(res)


if '__main__' == __name__:
    #fit_gru()
    gen_text()
