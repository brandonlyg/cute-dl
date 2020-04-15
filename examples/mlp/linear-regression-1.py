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
def fit_report(history, fpath, skip=0):
    plt.figure(figsize=(16, 12))
    #绘制训练历史
    loss = history['loss'][skip:]
    val_loss = history['val_loss'][skip:]
    x = history['steps'][skip:]
    #loss = history['loss'][400:]
    #val_loss = history['val_loss'][400:]

    plt.subplot(211)
    plt.plot(x, loss, 'b', label="Training loss")
    plt.plot(x, val_loss, 'r', label="Validation loss")
    y_max = max(max(val_loss), max(loss))
    y_min = min(min(val_loss), min(loss))
    txt = 'loss min:%f, final:%f\n'%(min(loss), loss[-1])
    txt = txt + 'val_loss min:%f, final:%f\n'%(min(val_loss), val_loss[-1])
    txt = txt + "steps:%d\n"%x[-1]
    txt = txt + "cost time:%fs\n"%history['cost_time']
    plt.text(x[-1], ((y_max-y_min) * 0.80)+y_min, txt,
            color='orange', horizontalalignment='right', verticalalignment='center',
            )
    #plt.text(x[-1]*0.99, max(val_loss) * 0.85,
    #        'val_loss min:%f, final:%f'%(min(val_loss), val_loss[-1]),
    #        color='orange', horizontalalignment='right', verticalalignment='center',
    #        )
    plt.xlabel("steps")
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
    #如果n次验证损失值没有进一步变小
    n = 10
    losses = history['val_loss']
    if len(losses) <= n:
        return

    losses = losses[10:]
    min_loss = min(losses)
    idx = losses.index(min_loss)
    if len(losses) - idx > n:
        sess.stop_fit()

report_path = "pics/linear-regression-1-report-"

'''
过拟合对比基准
'''
def fit0():
    print("fit0")
    model = Model([
        nn.Dense(128, inshape=1, activation='relu'),
        nn.Dense(256, activation='relu'),
        nn.Dense(1)
    ])
    model.assemble()

    sess = Session(model,
                loss=losses.Mse(),
                optimizer = optimizers.Fixed(),
            )

    history = sess.fit(ds, 200000, val_data=val_ds, val_epochs=1000,
                    listeners=[
                        FitListener('val_end', callback=lambda h:on_val_end(sess, h))
                        ]
                    )

    fit_report(history, report_path+'00.png', 10)

'''
使用L2正则化缓解过拟合
'''
def fit1():
    print("fit1")
    model = Model([
        nn.Dense(128, inshape=1, activation='relu'),
        nn.Dense(256, activation='relu'),
        nn.Dense(1)
    ])
    model.assemble()

    sess = Session(model,
                loss=losses.Mse(),
                optimizer = optimizers.Fixed(),
                genoptms = [optimizers.L2(0.00005)]
            )

    history = sess.fit(ds, 200000, val_data=val_ds, val_epochs=1000,
                    listeners=[
                        FitListener('val_end', callback=lambda h:on_val_end(sess, h))
                        ]
                    )
    fit_report(history, report_path+'01.png', 10)


'''
使用dropout缓解过拟合
'''
def fit2():
    print("fit2")
    model = Model([
        nn.Dense(128, inshape=1, activation='relu'),
        nn.Dense(256, activation='relu'),
        nn.Dropout(0.80),
        nn.Dense(1)
    ])
    model.assemble()

    sess = Session(model,
                loss=losses.Mse(),
                optimizer = optimizers.Fixed(),
            )

    history = sess.fit(ds, 200000, val_data=val_ds, val_epochs=1000,
                    listeners=[
                        FitListener('val_end', callback=lambda h:on_val_end(sess, h))
                        ]
                    )

    fit_report(history, report_path+'02.png', 15)


'''
同时使用L2和dropout缓解过拟合
'''
def fit3():
    print("fit3")
    model = Model([
        nn.Dense(128, inshape=1, activation='relu'),
        nn.Dense(64, activation='relu'),
        nn.Dropout(0.9),
        nn.Dense(1)
    ])
    model.assemble()

    sess = Session(model,
                loss=losses.Mse(),
                optimizer = optimizers.Fixed(),
                genoptms = [optimizers.L2(0.0005)]
            )

    history = sess.fit(ds, 200000, val_data=val_ds, val_epochs=1000,
                    listeners=[
                        FitListener('val_end', callback=lambda h:on_val_end(sess, h))
                        ]
                    )

    fit_report(history, report_path+'03.png', 15)



if __name__ == '__main__':
    #fit0()
    #fit1()
    fit2()
    #fit3()
    pass
