# coding=utf-8

import sys
sys.path.append("../..")
sys.path.append("../../cutedl")

'''
中国古诗词数据集
'''

import pdb
import os
import pickle
import glob
import re

import numpy as np
from cutedl.dataset import Dataset
from text_tool import Vocabulary

class ChinaPoetry:

    '''
    dir 数据集所在目录, 默认在./china_poetry目录下
    '''
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.dirname(__file__) + "/china_poetry"
            if dir == '/china_poetry':
                dir = '.'+dir

        self.__dir = dir
        self.__vocab = None


    @property
    def vocabulary(self):
        return self.__vocab

    '''
    加载数据集
    '''
    def load_ds(self, batch_size, seqlen):
        #读取原始数据
        print("read raw data")
        raw_data = self.__read_all()

        #加载词汇表
        fpath = self.__dir+"/vocabulary.pkl"
        vocab = None
        if not os.path.exists(fpath):
            vocab = self.__build_vocabulary(raw_data, fpath)
            self.__vocab = vocab
        else:
            vocab = Vocabulary.load(fpath)
            self.__vocab = vocab

        print("encoding")
        #筛选数据并编码
        datas = []
        labels = []
        for r in raw_data:
            if len(r) != seqlen:
                continue

            r = vocab.encode(sentence=r)
            datas.append(r[0:seqlen-1])
            labels.append(r[1:seqlen])

        datas = np.array(datas)
        labels = np.array(labels)

        print("load dataset successful")
        ds = Dataset(datas, labels, batch_size, drop_remainder=True)
        return ds

    '''
    构建词汇表
    '''
    def __build_vocabulary(self, raw_data, fpath):
        vocab = Vocabulary(start_index=0)
        for r in raw_data:
            vocab.update(sentence=r)

        print("vocab size:", vocab.size())
        #保存字典
        vocab.save(fpath)

        return vocab

    '''
    读取所有的数据
    '''
    def __read_all(self):
        res = []

        #读取单个文件
        def read_file(fname):
            sentences  = []
            txt = ''
            with open(fname, 'r') as f:
                txt = f.read()

            while txt.find('\n\n') > 0:
                end = txt.find('\n\n')
                #得到一首诗的数据
                item = txt[:end]
                txt = txt[end+2:]

                #去掉标题
                start = item.find('\n')
                item = item[start+1:]
                #去掉注释和空格
                item = re.sub(r'[\(（][^\)）]+[\)）]', '', item)
                item = re.sub(r'[》《“”:： ]', '', item)
                #把分割符替换成空格
                item = re.sub(r'[,，\.。；？！]', ' ', item)
                item = item.lstrip().rstrip()
                #分割成句子
                sts = item.split(' ')

                sentences += sts

            return sentences

        fnames = sorted(glob.glob(self.__dir+"/*.txt"))
        for fn in fnames:
            sentences = read_file(fn)
            res += sentences

        return res
