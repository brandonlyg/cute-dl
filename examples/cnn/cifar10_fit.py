# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../cutedl')

'''
cifar10数据集上的分类模型
'''
from datasets.cifar10 import Cifar10

cf10 = Cifar10('../datasets/cifar-10')
ds_train, ds_test = cf10.load(64)

from cutedl.model import Model
from cutedl.session import Session
from cutedl import session, losses, optimizers, utils
from cutedl import nn_layers as nn
from cutedl import cnn_layers as cnn

import fit_tools

report_path = "./pics/cifar10-fit-"
model_path = "./model/cifar10-fit-"

def fit(name, optimizer):
    inshape = ds_train.data.shape[1:]
    #pdb.set_trace()
    model = Model([
                cnn.Conv2D(32, (3,3), inshape=inshape),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                cnn.Conv2D(64, (3,3)),
                cnn.MaxPool2D((2,2), strides=(2,2)),
                cnn.Conv2D(64, (3, 3)),
                nn.Flatten(),
                nn.Dense(64),
                nn.Dropout(0.5),
                nn.Dense(10)
            ])
    model.assemble()

    sess = Session(model,
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizer
            )

    stop_fit = session.condition_callback(lambda :sess.stop_fit(), 'val_loss', 30)

    accuracy = lambda h: fit_tools.accuracy(sess, ds_test, h)

    def save_and_report(history):
        #pdb.set_trace()
        fit_tools.fit_report(history, report_path+name+".png")
        model.save(model_path+name)

    #pdb.set_trace()
    history = sess.fit(ds_train, 200, val_data=ds_test, val_steps=100,
                        listeners=[
                            stop_fit,
                            session.FitListener('val_end', callback=accuracy),
                            session.FitListener('val_end', callback=save_and_report)
                        ]
                    )

    save_and_report(history)


if '__main__' == __name__:
    fit('1', optimizers.Adam())
