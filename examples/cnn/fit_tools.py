# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

'''
生成模型的拟合报告
'''
def fit_report(history, fpath, skip=0):
    plt.figure(1, figsize=(16, 12))

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
    #pdb.set_trace()
    plt.text(x[-1], ((y_max-y_min) * 0.75)+y_min, txt,
            color='orange', horizontalalignment='right', verticalalignment='center',
            )

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(212)
    ymin = int(min(val_acc) * 100)/100
    ymax = ymin + (1-ymin)/0.75
    plt.ylim((ymin, ymax))
    #pdb.set_trace()
    plt.plot(x, val_acc, 'g', label="Validation accuracy")
    txt = 'val_accuracy max:%f, final:%f\n'%(max(val_acc), val_acc[-1])
    plt.text(x[-1], (ymax-ymin)*0.8 + ymin, txt,
            color='orange', horizontalalignment='right', verticalalignment='center',
            )
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.legend()

    plt.savefig(fpath)

    plt.clf()

def test_fitreport():

    steps = np.arange(10)
    val_loss = (np.arange(1, 11)*0.1)[::-1]
    loss = val_loss - 0.01

    start = 0.1
    end = 1.0
    val_accuracy = np.arange(0, 10)*0.1*(end-start) + start
    history = {
        'steps': steps.tolist(),
        'loss': loss.tolist(),
        'val_loss': val_loss.tolist(),
        'val_accuracy': val_accuracy.tolist(),
        'cost_time': 100
    }

    fit_report(history, "./pics/test.png")


'''
计算准确率
'''
def accuracy(sess, ds_test, history):
    val_pred = history['val_pred']
    y_pred = np.argmax(val_pred, axis=1)
    y_true = np.argmax(ds_test.label, axis=1)
    acc = (y_pred == y_true).astype(int).mean()

    if 'val_accuracy' not in history:
        history['val_accuracy'] = []

    val_accs = history['val_accuracy']
    val_accs.append(acc)
    print("accuracy: ", acc)
