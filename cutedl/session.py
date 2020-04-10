# coding=utf-8

import os
import pdb
import pickle
from model import Model
import reguls

'''
会话类型
实现模型的训练，保存，恢复训练状态
'''

class Session(object):

    '''
    model: Model对象
    loss: Loss对象
    optimizer: Optimizer对象
    '''
    def __init__(self, model, loss, optimizer):
        self.__model = model
        self.__loss = loss
        self.__optimizer = optimizer

    @property
    def model(self):
        return self.__model

    @property
    def loss(self):
        return self.__loss

    '''
    设置模型
    '''
    def set_model(self, model):
        self.__model = model

    '''
    分批训练
    '''
    def batch_train(self, data, label):
        #使用模型预测
        out = self.__model.predict(data, training=True)
        #使用损失函数评估误差
        loss = self.__loss(out, label)
        grad = self.__loss.gradient
        #pdb.set_trace()
        #反向传播梯度
        self.__model.backward(self.__loss.gradient)

        #正则化算法
        

        #更新模型参数
        self.__optimizer(self.__model)

        return loss


    '''
    保存session
    fpath: 保存的文件路径
        fpath+'.s.pkl' 是保存session的文件
        fpath+'.m.pkl' 是保存model的文件
    '''
    def save(self, fpath):
        model = self.__model
        self.__model = None

        model.save(fpath)

        realfp = fpath + ".s.pkl"
        with open(realfp, 'wb') as f:
            pickle.dump(self, f)

    '''
    加载session
    '''
    @classmethod
    def load(cls, fpath):
        realfp = fpath + ".s.pkl"
        if not os.path.exists(realfp):
            return None

        sess = None
        with open(realfp, 'rb') as f:
            sess = pickle.load(f)

        model = Model.load(fpath)
        sess.set_model(model)

        return sess
