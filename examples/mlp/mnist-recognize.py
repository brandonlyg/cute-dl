# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
手写数字识别模型
'''

from datasets.mnist import Mnist

'''
加载手写数字数据集
'''
mnist = Mnist('../datasets/mnist')
ds_train, ds_test = mnist.load(64, flatting=True)

import pdb
import numpy as np

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn

import matplotlib.pyplot as plt
'''
生成模型的拟合报告
'''
def fit_report(history, fpath, skip=0):
    plt.figure(figsize=(16, 12))
    #绘制训练历史
    loss = history['loss'][skip:]
    val_loss = history['val_loss'][skip:]
    val_acc = history['val_accuracy'][skip:]
    x = history['steps'][skip:]

    plt.subplot(211)
    plt.plot(x, loss, 'b', label="Training loss")
    plt.plot(x, val_loss, 'r', label="Validation loss")

    y_max = max(max(val_loss), max(loss))
    y_min = min(min(val_loss), min(loss))
    txt = 'loss min:%f, final:%f\n'%(min(loss), loss[-1])
    txt = txt + 'val_loss min:%f, final:%f\n'%(min(val_loss), val_loss[-1])
    txt = txt + "steps:%d\n"%x[-1]
    txt = txt + "cost time:%fs\n"%history['cost_time']
    plt.text(x[-1], ((y_max-y_min) * 0.75)+y_min, txt,
            color='orange', horizontalalignment='right', verticalalignment='center',
            )

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(212)
    plt.ylim((0.6, 1.2))
    plt.plot(x, val_acc, 'g', label="Validation accuracy")
    txt = 'val_accuracy max:%f, final:%f\n'%(max(val_acc), val_acc[-1])
    plt.text(x[-1], 1.1, txt,
            color='orange', horizontalalignment='right', verticalalignment='center',
            )
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.legend()

    plt.savefig(fpath)

'''
计算准确率
'''
def accuracy(sess, history):
    val_pred = history['val_pred']
    y_pred = np.argmax(val_pred, axis=1)
    y_true = np.argmax(ds_test.label, axis=1)
    acc = (y_pred == y_true).astype(int).mean()

    if 'val_accuracy' not in history:
        history['val_accuracy'] = []

    val_accs = history['val_accuracy']
    val_accs.append(acc)
    print("accuracy: ", acc)

    #连续20次大于91%停止训练
    count = len(val_accs)
    max = 20
    if count>max:
        stop = True
        for i in range(count-max, count):
            #pdb.set_trace()
            if val_accs[i] <= 0.91:
                stop = False
                break

        if stop:
            print("val accuracy >0.91 %d times. stop fit!"%max)
            sess.stop_fit()


report_path = "./pics/mnist-recoginze-"

'''
训练模型
'''
def fit(report_name, optimizer):
    inshape = ds_train.data.shape[1]
    model = Model([
                nn.Dense(10, inshape=inshape, activation='relu')
            ])
    model.assemble()

    sess = Session(model,
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizer
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 20)

    #pdb.set_trace()
    history = sess.fit(ds_train, 5000, val_epochs=1, val_data=ds_test,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=lambda h: accuracy(sess, h))
                        ]
                    )

    fit_report(history, report_path+report_name)

'''
固定学习率
'''
def fit0():
    lr = 0.0001
    print("fit1 lr:", lr)
    fit('0.png', optimizers.Fixed(lr))

def fit1():
    lr = 0.2
    print("fit0 lr:", lr)
    fit('1.png', optimizers.Fixed(lr))

def fit2():
    lr = 0.01
    print("fit2 lr:", lr)
    fit('2.png', optimizers.Fixed(lr))

'''
Momentum优化器
'''
def fit_use_momentum():
    lr = 0.002
    dpr = 0.9
    print("fit_use_momentum lr=%f, dpr:%f"%(lr, dpr))
    fit('momentum.png', optimizers.Momentum(lr, dpr))

'''
Adagrad优化器
'''
def fit_use_adagrad():
    lr = 0.001
    print("fit_use_adagrad lr=%f"%lr)
    fit('adagrad.png', optimizers.Adagrad(lr))

'''
RMSProp优化器
'''
def fit_use_rmsprop():
    sdpr = 0.99
    lr=0.0001
    print("fit_use_rmsprop lr=%f sdpr=%f"%(lr, sdpr))
    fit('rmsprop.png', optimizers.RMSProp(lr, sdpr))

'''
Adadelta优化器
'''
def fit_use_adadelta():
    dpr = 0.99
    print("fit_use_adadelta dpr=%f"%dpr)
    fit('adadelta.png', optimizers.Adadelta(dpr))

'''
Adam优化器
'''
def fit_use_adam():
    lr = 0.0001
    mdpr = 0.9
    sdpr = 0.99
    print("fit_use_adam lr=%f, mdpr=%f, sdpr=%f"%(lr, mdpr, sdpr))
    fit('adam.png', optimizers.Adam(lr, mdpr, sdpr))


if '__main__' == __name__:
    fit0()
    #fit1()
    #fit2()

    #fit_use_momentum()
    #fit_use_adagrad()
    #fit_use_rmsprop()
    #fit_use_adadelta()
    #fit_use_adam()

    pass
