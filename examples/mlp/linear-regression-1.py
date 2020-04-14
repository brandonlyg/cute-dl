# coding=utf-8

import sys
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
1. 验证过拟合解决方案
2. 使用fit方法训练模型
'''

import matplotlib.pyplot as plt

from cutedl.model import Model
from cutedl.session import Session, FitListener
from cutedl import nn_layers as nn
from cutedl import losses, optimizers
from cutedl.dataset import Dataset

import ds_generator


#生成Dataset数据集
def gen_dataset():
    ds_1 = ds_generator.ds_1
    ds = Dataset(ds_1['train_x'], ds_1['train_y'], 16)

    val_ds = Dataset(ds_1['test_x'], ds_1['test_y'], 40)

    return ds, val_ds

ds, val_ds = gen_dataset()

'''
生成模型的拟合报告
'''
def fit_report(history, fpath):
    plt.figure(figsize=(16, 12))
    #绘制训练历史
    loss = history['loss']
    val_loss = history['val_loss']
    x = history['steps']
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
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()

    #pdb.set_trace()
    #绘制拟合图形
    x = val_ds.data.reshape(-1)
    y = val_ds.label.reshape(-1)
    y_pred = history['val_pred'].reshape(-1)
    plt.subplot(212)
    plt.scatter(x, y)
    plt.plot(x, y_pred, 'r')
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend()

    plt.savefig(fpath)

def on_val_end(sess, history):
    #如果100次验证损失值没有进一步变小
    losses = history['val_loss']
    min_loss = min(losses)
    idx = losses.index(min_loss)
    if len(losses) - idx > 100:
        sess.stop_fit()

report_path = "pics/linear-regression-1-report-"

#使用L2正则化缓解过拟合
def fit1():
    model = Model([
        nn.Dense(128, inshape=1, activation='relu'),
        nn.Dense(64, activation='relu'),
        nn.Dense(1)
    ])
    model.assemble()

    sess = Session(model,
                loss=losses.Mse(),
                optimizer = optimizers.Fixed()
                #genoptms = [optimizers.L2(0.1)]
            )

    history = sess.fit(ds, 30000, val_data=val_ds, val_epochs=1000,
                    listeners=[
                        FitListener('val_end', callback=lambda h:on_val_end(sess, h))
                        ]
                    )
    fit_report(history, report_path+'01.png')


if __name__ == '__main__':
    fit1()

    pass
