# coding=utf-8

import sys
sys.path.append('../..')
sys.path.append('../../cutedl')

import os
import pickle
import pdb
import time
import numpy as np
import matplotlib.pyplot as plt

from cutedl.model import Model
from cutedl.session import Session
from cutedl import nn_layers as nnlys
from cutedl import losses, optimizers

import ds_generator

ds_0 = ds_generator.ds_0
ds_1 = ds_generator.ds_1


'''
使用MLP(Multi-Layer Perceptron)模型实现广义线性回归, 该模型的任务是拟合一个函数
'''

model_path = "./model/mlp_linear_regression"

'''
训练模型
'''
def train(epochs, ds, model=None, batch_size=64, record_epochs=1):
    #加载/构建session
    sess = None
    if model is None:
        sess = Session.load(model_path)
    else:
        sess = Session(model,
                    loss=losses.Mse(),
                    optimizer = optimizers.Fixed()
                )

    train_x = ds['train_x']
    train_y = ds['train_y']
    test_x = ds['test_x']
    test_y = ds['test_y']

    batchs = int(train_x.shape[0]/batch_size)
    print("epochs:%d, batchs=%d"%(epochs, batchs))

    #记录训练历史
    history = {
        'loss': [],
        'val_loss': [],
        'epochs': [],
        'val_x': test_x,
        'val_y': test_y,
        'val_pred': None
    }

    print("start training ")
    t_start = time.time()
    steps = epochs * batchs

    epoch = 1
    #循环训练
    for step in range(steps):
        start = (step % batchs) * batch_size
        end = start + batch_size
        batch_x = train_x[start:end]
        batch_y = train_y[start:end]

        loss = sess.batch_train(batch_x, batch_y)

        cur_epoch = int(step/batchs) + 1

        #每轮打印一次
        if step > 0 and  step % batchs == 0:
            print((('epoch:%05d/%d loss=%f'%(cur_epoch, epochs, loss))+' '*50)[:50], end='\r')

        #记录
        if step % batchs == 0 and (cur_epoch - epoch == record_epochs or cur_epoch == epochs):
            epoch = cur_epoch

            y_pred = sess.model.predict(test_x)
            val_loss = sess.loss(test_y, y_pred)

            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            history['epochs'].append(epoch)
            history['val_pred']  = y_pred

            print((('epoch:%05d/%d loss=%f, val_loss=%f'%(cur_epoch, epochs, loss, val_loss))+' '*50)[:50], end='\r')
            print("")

    sess.save(model_path)
    print("training finished cost:%f" % (time.time() - t_start))

    return history


'''
生成模型的拟合报告
'''
def fit_report(history, fpath):
    plt.figure(figsize=(16, 12))
    #绘制训练历史
    loss = history['loss']
    val_loss = history['val_loss']
    x = history['epochs']
    #loss = history['loss'][400:]
    #val_loss = history['val_loss'][400:]

    plt.subplot(211)
    plt.plot(x, loss, 'b', label="Training loss")
    plt.plot(x, val_loss, 'r', label="Validation loss")
    y_max = max(max(val_loss), max(loss))
    y_min = min(min(val_loss), min(loss))
    plt.text(x[-1], ((y_max-y_min) * 0.80)+y_min,
            'loss min:%f, final:%f\nval_loss min:%f, final:%f'%(min(loss), loss[-1], min(val_loss), val_loss[-1]),
            color='orange', horizontalalignment='right', verticalalignment='center',
            )
    #plt.text(x[-1]*0.99, max(val_loss) * 0.85,
    #        'val_loss min:%f, final:%f'%(min(val_loss), val_loss[-1]),
    #        color='orange', horizontalalignment='right', verticalalignment='center',
    #        )
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    #pdb.set_trace()
    #绘制拟合图形
    x = history['val_x'].reshape(-1)
    y = history['val_y'].reshape(-1)
    y_pred = history['val_pred'].reshape(-1)
    plt.subplot(212)
    plt.scatter(x, y)
    plt.plot(x, y_pred, 'r')
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend()

    plt.savefig(fpath)


report_path = "pics/linear-regression-report-"

#欠拟合示例
def fit_1():
    model = Model([
        nnlys.Dense(32, inshape=1),
        nnlys.Dense(1)
    ])
    model.assemble()
    #这个模型是一个线性模型, 用来拟合非线性函数, 模型复杂度不够，一定会表现出欠拟合
    history = train(20000, ds_0, model, record_epochs=100)
    fit_report(history, report_path+'01.png')

#使用增加模型复杂度解决欠拟合问题
def fit_2():
    model = Model([
        nnlys.Dense(32, inshape=1, activation='relu'),
        nnlys.Dense(1)
    ])
    model.assemble()
    #使用了relu激活函数模型变成了非线性的, 增加了模型的复杂度
    history = train(30000, ds_0, model, record_epochs=300)
    history['loss'] = history['loss'][5:]
    history['val_loss'] = history['val_loss'][5:]
    history['epochs'] = history['epochs'][5:]
    fit_report(history, report_path+'02.png')

#过拟合
def fit_3():
    model = Model([
        nnlys.Dense(512, inshape=1, activation='relu'),
        nnlys.Dense(128, activation='relu'),
        nnlys.Dense(1)
    ])
    model.assemble()

    history = train(30000, ds_1, model, batch_size=16, record_epochs=300)
    history['loss'] = history['loss'][20:]
    history['val_loss'] = history['val_loss'][20:]
    history['epochs'] = history['epochs'][20:]
    fit_report(history, report_path+'03.png')

#减少参数数量缓解过拟合
def fit_4():
    model = Model([
        nnlys.Dense(128, inshape=1, activation='relu'),
        nnlys.Dense(64, activation='relu'),
        nnlys.Dense(1)
    ])
    model.assemble()

    history = train(30000, ds_1, model, batch_size=16, record_epochs=300)
    history['loss'] = history['loss'][20:]
    history['val_loss'] = history['val_loss'][20:]
    history['epochs'] = history['epochs'][20:]
    fit_report(history, report_path+'04.png')

if '__main__' == __name__:
    #fit_1()
    #fit_2()

    #fit_3()
    fit_4()

    pass
