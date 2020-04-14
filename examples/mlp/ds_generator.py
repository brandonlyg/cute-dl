# coding=utf-8

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

'''
数据集生成器
'''

'''
任务目标函数
'''
def target_func(x):
    #加入服从参数(0, 0.25^2)正态分布噪声
    y = (x - 2)**2 + 0.25 * np.random.randn(len(x))
    return y

'''
生成数据集
返回: train_x, train_y, test_x, test_y
train_x, train_y 训练数据集的数据和标签
test_x, test_y 验证数据解的数据和标签
'''
def generate_dataset():
    '''
    生成200条数据, 随机取出80%条作为训练数据集, 剩余数据为测试数据集
    '''
    fpath = "./ds.pkl"
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            ds = pickle.load(f)
            return ds

    count = 200
    x = np.linspace(-1, 5, count)
    y = target_func(x)

    #打乱顺序
    indices = np.arange(count)
    np.random.shuffle(indices)
    #训练数据集
    split = int(count*0.8)
    idxs = indices[:split]
    train_x = x[idxs].reshape((-1,1))
    train_y = y[idxs].reshape((-1,1))

    #测试数据集
    idxs = sorted(indices[split:])
    test_x = x[idxs].reshape((-1, 1))
    shape = test_x.shape
    test_y = y[idxs].reshape((-1, 1))

    ds = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y
    }
    with open(fpath, 'wb') as f:
        pickle.dump(ds, f)

    return ds

#得到数据集
ds_0 = generate_dataset()
print("train shape:", ds_0['train_x'].shape)
print("test shape:", ds_0['test_x'].shape)

#训练集只取一部分
ds_1 = {
    'train_x': ds_0['train_x'][:16],
    'train_y': ds_0['train_y'][:16],
    'test_x': ds_0['test_x'],
    'test_y': ds_0['test_y']
}

'''
看一下部分测试数据集的图形
'''
def show_testds():
    test_x = ds_0['test_x']
    test_y = ds_0['test_y']

    plt.figure(figsize=(12,8))
    x = test_x.reshape(-1)
    y = test_y.reshape(-1)
    plt.scatter(x, y)
    plt.text(max(x)*0.95, max(y)*0.95, 'test ds', color='orange',
            horizontalalignment='right', verticalalignment='center'
            )
    plt.savefig("./pics/testds.png")

#show_testds()
