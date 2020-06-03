# coding=utf-8

import os
import pickle

'''
文本预处理工具
'''

'''
词汇表类
'''
class Vocabulary(object):

    def __init__(self):
        self.__idx2w = [] #索引到词的映射
        self.__w2idx = {} #词到索引的映射

        self.__add_buf = set()

    '''
    添加词汇
    '''
    def add(self, words=[], sentence=""):
        self.__add_buf.update(words)
        if len(sentence) > 0:
            self.__add_buf.update(sentence)

    '''
    更新词汇表
    '''
    def update(self, words=[], sentence=""):
        self.__add_buf.update(words)
        if len(sentence) > 0:
            self.__add_buf.update(sentence)

        words = sorted(list(self.__add_buf))
        self.__add_buf = set()

        idx = len(self.__idx2w)
        for w in words:
            if w in self.__w2idx:
                continue

            self.__idx2w.append(w)
            self.__w2idx[w] = idx+1
            idx += 1

    '''
    把词序列编码成整数序列
    '''
    def encode(self, words=[], sentence="", drop_missing=False):
        src = words
        for i in range(len(sentence)):
            w = sentence[i]
            src.append(w)

        res = []
        missing = len(self.__idx2w) + 1
        for w in src:
            ec = missing
            if w in self.__w2idx:
                ec = self.__w2idx[w]

            if ec != missing:
                res.append(ec)
            elif not drop_missing:
                res.append(ec)

        return res

    '''
    把整数序列解码成词序列
    '''
    def decode(self, intseq):
        res = []
        missing = len(self.__idx2w) + 1
        for ec in intseq:
            if ec >= missing or ec <= 0:
                res.append(' ')
            else:
                res.append(self.__idx2w[ec-1])

        return res

    def size(self):
        return len(self.__idx2w)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        vocab = None
        with open(path, 'rb') as f:
            vocab = pickle.load(f)

        return vocab

'''
根据给定文本, 生成或扩充字符词典, 词典只包含一个字符的词
return 新的词典
'''
def gen_character_vocabulary(text_file_path, vocabulary=None):
    txt = None
    with open(text_file_path, 'r') as f:
        txt = f.read()

    if vocabulary is None:
        vocabulary = Vocabulary()

    vocabulary.update(sentence=txt)

    return vocabulary
