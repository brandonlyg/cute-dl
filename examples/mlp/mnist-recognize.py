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
    plt.ylim((0, 1.3))
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
def accuracy(history):
    val_pred = history['val_pred']
    y_pred = np.argmax(val_pred, axis=1)
    y_true = np.argmax(ds_test.label, axis=1)
    acc = (y_pred == y_true).astype(int).mean()

    if 'val_accuracy' not in history:
        history['val_accuracy'] = []

    history['val_accuracy'].append(acc)
    print("\naccuracy: ", acc, end='\r')


report_path = "./pics/mnist-recoginze-"

'''
训练模型
'''
def fit():
    inshape = ds_train.data.shape[1]
    model = Model([
                nn.Dense(10, inshape=inshape, activation='relu')
            ])
    model.assemble()

    sess = Session(model,
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.Fixed(0.001)
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 10)

    #pdb.set_trace()
    history = sess.fit(ds_train, 20000, val_epochs=5, val_data=ds_test,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=accuracy)
                        ]
                    )

    fit_report(history, report_path+"0.png")


if '__main__' == __name__:
    fit()

    pass
