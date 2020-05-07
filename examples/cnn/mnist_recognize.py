# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
使用卷积神经网络实现的手写数字识别模型
'''

from datasets.mnist import Mnist

'''
加载手写数字数据集
'''
mnist = Mnist('../datasets/mnist')
ds_train, ds_test = mnist.load(64)

import pdb
import numpy as np

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import cnn_layers as cnn

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
    plt.ylim((0.3, 1.2))
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
model_path = "./model/mnist-recoginze-"

'''
训练模型
'''
def fit01(report_name, model_name, optimizer):
    inshape = ds_train.data.shape[1:]
    #pdb.set_trace()
    model = Model([
                cnn.Conv2D(32, (3,3), inshape=inshape),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                cnn.Conv2D(64, (3,3)),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                nn.Flatten(),
                nn.Dense(1024),
                nn.Dense(10)
            ])
    model.assemble()

    sess = Session(model,
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizer
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 20)

    #pdb.set_trace()
    history = sess.fit(ds_train, 200, val_data=ds_test, val_steps=100,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=lambda h: accuracy(sess, h))
                        ]
                    )

    fit_report(history, report_path+report_name)

    model.save(model_name)


if '__main__' == __name__:
    fit01('1.png', model_path+"1", optimizers.Adam())
