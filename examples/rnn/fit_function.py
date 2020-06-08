# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../datasets')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
拟合一个函数
'''

import pdb
import numpy as np
from cutedl.dataset import Dataset
from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import rnn_layers as rnn
from datasets import fit_tools

'''
从函数采用数据集
seqlen 序列长度
batch_size 批数据量
batches 批次数
'''
def sample_dataset(seqlen, batch_size, batches):

    #采样函数
    def sample_function(x):
        y = 3*np.sin(2 * x * np.pi) + np.cos(x * np.pi) + np.random.uniform(-0.05,0.05,len(x))
        return y

    def build_dataset(seq):
        datas = []
        labels = []
        #pdb.set_trace()
        for i in range(len(seq) - seqlen):
            d = seq[i:i+seqlen]
            datas.append(d[0:seqlen-1])
            labels.append(d[-1])

        datas = np.array(datas)
        m, t = datas.shape
        datas = datas.reshape((m, t, 1))
        labels = np.array(labels).reshape((-1, 1))
        ds = Dataset(datas, labels, batch_size)
        return ds


    #样本总数
    total = batch_size * batches

    #采用序列总长度
    length = seqlen + total

    #采样频率
    SAMPLE_RATE=0.01
    #采样开始位置
    SAMPLE_START = 1

    #采样训练数据集
    end = SAMPLE_START + SAMPLE_RATE * length
    x =  np.linspace(SAMPLE_START, end, num=length)
    y = sample_function(x)
    train = build_dataset(y)

    #采样测试数据集
    start = end+SAMPLE_RATE
    count = int(length * 0.2)
    end = start + SAMPLE_RATE * count
    x = np.linspace(start, end, count)
    y = sample_function(x)
    test = build_dataset(y)

    return train, test

'''
训练集采样区间: [1, 200.01)
测试集采样区间: [200.02, 240.002)
'''
ds_train, ds_test = sample_dataset(100, 32, 200)
#pdb.set_trace()

report_path = "./pics/fit-function-"
model_path = "./model/fit-function-"

def fit(name, model):
    sess = Session(model,
                loss = losses.Mse(),
                optimizer = optimizers.Adam()
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 20)


    def save_and_report(history):
        #pdb.set_trace()
        val_true = history['val_true']
        m,_ = val_true.shape
        val_x = np.linspace(1, 1+m*0.01, m)
        history['val_x'] = val_x
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 100, val_data=ds_test,
                        listeners=[
                            stop_fit,
                            session.FitListener('epoch_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


def fit_gru():
    model = Model([
                rnn.GRU(32, 1),
                nn.Filter(),
                nn.Dense(32),
                nn.Dense(1, activation='linear')
            ])
    model.assemble()
    fit('gru', model)


def fit_lstm():
    model = Model([
                rnn.LSTM(32, 1),
                nn.Filter(),
                nn.Dense(32),
                nn.Dense(1, activation='linear')
            ])
    model.assemble()
    fit('lstm', model)

if '__main__' == __name__:
    #fit_gru()
    fit_lstm()
    pass
