# coding=utf-8


import sys
sys.path.append("../..")
sys.path.append("../../cutedl")

'''
imdb-review 数据集工具
'''

import pdb
import glob
import pickle
import os
import re

import numpy as np
from cutedl.dataset import RaggedDataset
from text_tool import Vocabulary

class IMDBR:
    '''
    dir 数据集所在目录, 默认在./imdbr目录下
    '''
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.dirname(__file__) + "/imdbr"
            if dir == '/imdbr':
                dir = '.'+dir

        self.__dir = dir

    def load_ds(self, batch_size, batches=0):
        #默认加imdbr文件
        ds_file = self.__dir + "/imdbr.pkl"
        if not os.path.exists(ds_file):
            #如果没有, 预处理文本文件，生成imdbr.pkl
            self.__pre_process()

        ds = None
        print("load dataset ...")
        with open(ds_file, 'rb') as f:
            ds = pickle.load(f)

        #词汇量
        vocab_size = 0
        with open(self.__dir+"/vocab_size.txt", 'r') as f:
            str = f.read()
            vocab_size = int(str)

        #pdb.set_trace()
        '''
        ds 格式
        {
        b'train_datas':[],
        b'train_labels': [],
        b'test_datas': [],
        b'test_labels': []
        }
        '''
        print("build dataset. train data count:%d, test data count:%d, vocab_size:%d" % (
                len(ds['train_datas']), len(ds['test_datas']), vocab_size)
             )

        labels = np.array(ds['train_labels']).reshape((-1, 1))
        train = RaggedDataset(ds['train_datas'], labels, batch_size, batches)

        labels = np.array(ds['test_labels']).reshape((-1, 1))
        test = RaggedDataset(ds['test_datas'], labels, batch_size, batches)

        return train, test, vocab_size

    '''
    预处理数据集
    '''
    def __pre_process(self):
        #数据集信息
        pdir = self.__dir + "/raw"
        dsinfos = [
            {'dir': pdir+"/train/neg", 'ds':'train', 'label': 0},
            {'dir': pdir+"/train/pos", 'ds':'train', 'label': 1},
            {'dir': pdir+"/test/neg", 'ds':'test', 'label': 0},
            {'dir': pdir+"/test/pos", 'ds':'test', 'label': 1}
        ]

        ds = {
            'train_datas': [],
            'train_labels': [],
            'test_datas': [],
            'test_labels': []
        }
        #加载词典
        vocab = Vocabulary()
        with open(pdir+"/imdb.vocab") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                w = lines[i]
                lines[i] = w[0:-1]
            #pdb.set_trace()
            vocab.update(words=lines)

        #把文本内容转换成词序列
        def to_words(txt):
            #展开缩写
            txt = re.sub(r'\'s ', ' is ', txt)
            txt = re.sub(r'n\'t ', ' not ', txt)
            #过滤掉html tag
            txt = re.sub(r'<[^>]*/?[\w]*>', ' ', txt)
            #过滤掉标点符号
            txt = re.sub(r'[^a-zA-Z]', ' ', txt)
            #全部转换成小写
            txt = txt.lower()

            words = re.split(r' +', txt)
            return words

        #加载一条dsi中的数据
        def load_by_dsi(dsi):
            datas = []
            dir = dsi['dir']+"/*.txt"
            fnames = glob.glob(dir)
            print("dir: %s, file count:%d" % (dir, len(fnames)))
            count = 0
            for fn in fnames:
                with open(fn, 'r') as f:
                    txt = f.read()
                    words = to_words(txt)
                    datas.append(words)
                    count += 1

                    readinfo = "read file %d/%d"%(count, len(fnames)) + " "*50
                    print(readinfo[:50], end='\r')
            print("")
            n = len(datas)
            labels = [dsi['label']] * n
            return datas, labels

        #读取数据集并编码
        for dsi in dsinfos:
            datas, labels = load_by_dsi(dsi)
            #pdb.set_trace()
            #编码
            for i in range(len(datas)):
                d = datas[i]
                datas[i] = vocab.encode(words=d, drop_missing=True)

            kd = dsi['ds'] + "_datas"
            kl = dsi['ds'] + "_labels"
            ds[kd] = ds[kd] + datas
            ds[kl] = ds[kl] + labels

        #pdb.set_trace()
        with open(self.__dir+"/imdbr.pkl", 'wb') as f:
            pickle.dump(ds, f)


if '__main__' == __name__:
    imdbr = IMDBR()
    train, test, vocab_size = imdbr.load_ds(64)
    print("train batch count:%d, test batch_count:%d" % (train.batch_count, test.batch_count))

    #pdb.set_trace()
    for x, y in train.as_iterator():
        print("x shape: ", x.shape, " y shape: ", y.shape)
        break

    for x, y in test.as_iterator():
        print("x shape: ", x.shape, " y shape: ", y.shape)
        break
