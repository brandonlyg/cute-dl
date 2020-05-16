# coding=utf-8

import os
import pdb
import pickle
import time
import numpy as np

from model import Model

'''
会话类型
实现模型的训练，保存，恢复训练状态
'''

class Session(object):

    '''
    model: Model对象
    loss: Loss对象
    optimizer: Optimizer对象, 学习率优化器
    genoptms: list[Optimizer]对象, 广义参数优化器列表,
                    列表中的优化器将会在optimizer之前按顺序执行
    '''
    def __init__(self, model, loss, optimizer, genoptms=None):
        self.__model = model
        self.__loss = loss
        self.__optimizer = optimizer
        self.__genoptms = genoptms

        #拟合开关, 关闭时fit终止
        self.__fit_switch = True

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
        y_pred = self.__model.predict(data, training=True)
        #使用损失函数评估误差
        loss = self.__loss(label, y_pred)
        grad = self.__loss.gradient
        #pdb.set_trace()
        #反向传播梯度
        self.__model.backward(self.__loss.gradient)

        #执行广义优化器更新参数
        if self.__genoptms is not None:
            for optm in self.__genoptms:
                optm(self.__model)

        #执行学习率优化器更新参数
        self.__optimizer(self.__model)

        return loss

    #停止训练
    def stop_fit(self):
        self.__fit_switch = False

    '''
    自动分批训练拟合模型
    data  训练数据集 Dataset对象
    epochs 训练轮数
    val_data 验证数据集 Dataset对象
    val_epochs 每val_epochs步使用val_data进行一次验证, 同时记录记录当前训练的损失值.
    val_steps 和val_epochs的作用一样, 如果>0优先使用这个设置.
    listeners 训练事件监听器. FitLisener类型.
              训练过程产生的事件有:
              epoch_start  每轮训练开始触发
              epoch_end    每轮训练结束触发
              val_start    每次验证开始触发
              val_end      每次验证结束触发

    return 训练历史记录history
        history格式:
        {
            'loss': [],
            'val_loss': [],
            'steps': [],
            'val_pred': darray,
            'cost_time': float
        }
    '''
    def fit(self, data, epochs, val_data=None, val_epochs=1, val_steps=0, listeners=[]):
        history = {
            'loss': [],
            'val_loss': [],
            'steps': [],
            'val_pred': None,
            'cost_time': 0
        }
        self.__fit_switch = True

        start_time = time.time()

        if val_data is None:
            history['val_loss'] = None

        if val_epochs <= 0 or val_epochs >= epochs:
            val_epochs = 1

        if val_steps <= 0:
            val_steps = val_epochs * data.batch_count

        print("val_steps: ", val_steps)

        #事件派发
        def event_dispatch(event):
            #pdb.set_trace()
            for listener in listeners:
                listener(event, history)

        #使用验证数据集验证
        def validation():
            if val_data is None:
                return None, None

            val_pred = None
            losses = []
            for batch_x, batch_y in val_data.as_iterator():
                #pdb.set_trace()
                y_pred = self.__model.predict(batch_x)
                loss = self.__loss(batch_y, y_pred)
                losses.append(loss)

                if val_pred is None:
                    val_pred = y_pred
                else:
                    val_pred = np.vstack((val_pred, y_pred))

            loss = np.mean(np.array(losses))
            return loss, val_pred

        #记录
        def record(loss, val_loss, val_pred, step):
            history['loss'].append(loss)
            history['steps'].append(step)
            history['cost_time'] = time.time() - start_time

            if history['val_loss'] is not None and val_loss is not None :
                history['val_loss'].append(val_loss)
                history['val_pred'] = val_pred

        #显示训练进度
        def display_progress(epoch, epochs, step, steps, loss, val_loss=-1):
            prog = (step % steps)/steps
            w = 20

            str_epochs = ("%0"+str(len(str(epochs)))+"d/%d")%(epoch, epochs)

            txt = (">"*(int(prog * w))) + (" "*w)
            txt = txt[:w]
            if val_loss < 0:
                txt = txt + (" loss=%f   "%loss)
                print("%s %s"%(str_epochs, txt), end='\r')
            else:
                txt = "loss=%f, val_loss=%f%s"%(loss, val_loss, ''*w)
                #print("")
                print("%s %s"%(str_epochs, txt))


        #开始训练
        step = 0
        for epoch in range(epochs):
            if not self.__fit_switch:
                break

            event_dispatch("epoch_start")
            data.shuffle()
            for batch_x, batch_y in data.as_iterator():
                if not self.__fit_switch:
                    break
                #pdb.set_trace()
                loss = self.batch_train(batch_x, batch_y)
                step += 1
                if step % val_steps == 0:
                    event_dispatch("val_start")
                    val_loss, val_pred = validation()
                    record(loss, val_loss, val_pred, step)
                    display_progress(epoch+1, epochs, step, val_steps, loss, val_loss)
                    event_dispatch("val_end")

                    print("")
                else:
                    display_progress(epoch+1, epochs, step, val_steps, loss)

            event_dispatch("epoch_end")

        history['cost_time'] = time.time() - start_time

        return history

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


'''
拟合过程监听器
'''
class FitListener(object):

    '''
    events  监听事件的列表
    kargs:
        callback 回调函数
    '''
    def __init__(self, *events, **kargs):
        self.__events = set(events)

        if 'callback' not in kargs:
            raise Exception("miss param 'callback'")
        self.__callback = kargs['callback']

    def __call__(self, event, history):
        if event not in self.__events:
            return

        #pdb.set_trace()
        self.__callback(history)


'''
条件回调监听器
func    满足条件后的回调函数
key     监视key, listener会监视history[key]的值
times   当history[key]属性连续times次没有得到更小的值时停止训练
'''
def condition_callback(func, key, times, event='val_end'):
    def cb(func, history):
        losses = history[key]
        if len(losses) <= times:
            return

        losses = losses[times:]

        min_loss = min(losses)
        min_idx = losses.index(min_loss)

        stop = False
        if len(losses) - min_idx > times:
            #已连续times次监视值没有更小值
            stop = True
        elif min_idx == len(losses) - 1 and min_idx + 1 >= times:
            start = losses[min_idx+1-times]
            if start - min_loss < 0.001: #减少的太少也可认为没有减少
                stop = True

        if stop:
            func()

    listener = FitListener(event, callback=lambda h:cb(func, h))

    return listener
