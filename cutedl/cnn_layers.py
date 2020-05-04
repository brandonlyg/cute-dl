# coding=utf-8

'''
卷积层实现
'''
import copy
import numpy as np
from model import Layer, LayerParam
import utils

'''
2d 卷积层实现
输入形状 (m, c, (h, w))
输出性质 (m, c_, (h_, w_))
c 输入通道数
c_ 输出通道数
h, h_ 输入输出特征图的高度
w, w_ 输入输出特征图的宽度

k 卷积核大小, 卷积核的形状为: (kh, kw)
s 卷积运算步长 (sh, sw)
ph 高度填充
pw 宽度填充

h_ = (h + 2*ph - kh + 1)/sh
w_ = (w + 2*pw - kw + 1)/sw
不填充 ph=pw=0
填充到原来的大小: h_=h w_=w
ph = (sh*h - h + kh -1)/2, 当sh=1时 ph = (kh-1)/2  k是奇数
pw = (sw*w - w + kw -1)/2, 当sw=1时 pw = (kw-1)/2  k是奇数
'''
def Conv2D(Layer):
    tag='Conv2D'

    '''
    channels 输出通道数 int
    kernel_size 卷积核形状 (kh, kw)
    strids  卷积运算步长(sh, sw)
    padding 填充方式 'valid': 步填充. 'same': 使输出特征图和输入特征图形状相同
    inshape 输入形状 (c, h, w)
            c 输入通道数
            h 特征图高度
            w 特征图宽度
    '''
    def __init__(self, channels, kernel_size, strids=(1,1), padding='same', inshape=None, activation='relu'):
        self.__ks = kernel_size
        self.__st = strids
        self.__pd = (0, 0)
        self.__padding = padding

        #参数
        self.__W = None #(c*kernel_size, c_)
        self.__b = None #(c_)

        #输入输出形状
        self.__inshape = (-1, -1, -1)
        self.__outshape = None

        #真实输出形状
        self.__real_outshape = None

        #输出形状
        outshape = self.check_shape(channels)
        if outshape is None:
            raise Exception("invalid outshape: "+str(channels))

        self.__outshape = outshape

        #输入形状
        if inshape is not None:
            self.__inshape = self.check_shape(inshape)
            if self.__in_shape is None:
                raise Exception("invalid inshape: "+str(inshape))

            self.__outshape = self.__compute_outshape(self.__inshape)

        super().__init__(activation)

        self.__in_batch_shape = None
        self.__in_batch = None

    '''
    计算输出形状
    '''
    def __compute_outshape(self, inshape):
        _, h, w = inshape
        kh, kw = self.__ks
        sh, sw = self.__st

        #根据输入形状计算输出形状
        h_ = -1
        w_ = -1
        if 'same' == self.__padding:
            _, h_, w_ = inshape
            self.__pd = ((h_*sh - h + kh - 1)//2, (w_*sw - w + kw - 1)//2)
        elif 'valid' == self.__padding:
            _, h, w = inshape
            kh, kw = self.__ks
            sh, sw = self.__st
            h_ = (h - kh + 1)/sh
            w_ = (w - kw + 1)/sw
            self.__pd = (0, 0)
        else:
            raise Exception("invalid padding: "+self.__padding)

        outshape = self.__outshape + (h_, w_)
        return outshape

    @property
    def inshape(self, inshape):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    @property
    def params(self):
        return [self.__W, self.__b]

    def init_params(self):
        inshape = self.inshape
        outshape = self.__real_outshape
        in_chnls = inshape[0]
        out_chnls = outshape[0]
        k = self.__ks

        #展平形状(c*kh*kw, c_), 把卷积运算转换成矩阵运算, 优化性能
        shape = (in_chnls * utils.flat_shape(k), out_chnls)
        std = 0.01
        wval = np.random.randn(shape[0], shape[1]) * std
        bval = np.random.randn(shape[1]) * std

        W = LayerParam(self.name, 'weight', wval)
        b = LayerParam(self.name, 'bias', bval)

        self.__W = W
        self.__b = b

    def set_prev(self, prev_layer):
        inshape = prev_layer.outshape
        self.__outshape = self.__compute_outshape(inshape)
        self.__real_outshape = self.__outshape
        self.__inshape = inshape

        super().set_prev(pre_layer)

    def set_next(self, next_layer):
        if len(next_layer.inshape) < len(self.outshape):
            #适应下一层的输入要求
            l = len(next_layer.inshape)
            outshape = list(self.outshape[0:l])
            tmp = self.outshape[l-1:]
            outshape[-1] = utils.flat_shape(tmp)
            self.__outshape = tuple(outshape)

        super().set_next(next_layer)

    '''
    把特征图转换成方便卷积运算矩阵
    '''
    def __img2mat(self, in):
        ph, pw = self.__pad
        m, c, h, w = in.shape
        #先填充
        padded = np.zeros((m, c, h + 2*ph, w + 2*pw))
        padded[:, :, ph:(ph+h), pw:(pw+w)] = in

        c_, h_, w_ = self.__real_outshape
        kh,kw = self.__ks
        #转换成矩阵(m, h_, w_, c*kh*kw)
        out = np.zeros((m, h_, w_, c*kh*kw))
        for i in range(h_):
            for j in range(w_):
                #(m, c, kh, kw)
                cov = padded[:, :, i*kh:(i+1)*kh, j*kw:(1+j)kw]
                #(m, c*kh*kw)
                cov = cov.reshape((m, c*kh*kw))
                out[:, i, j, :] = cov

        #转换成(m*h_*w_, c*kh*kw)
        out = out.reshape((m*h_*w_, c*kh*kw))

        return out

    '''
    把卷积矩阵还原成特征图
    '''
    def __mat2img(self, mat):
        c_, h_, w_ = self.__real_outshape
        m, c, h, w = self.__in_batch_shape
        kh, kw = self.ks
        ph, pw = self.__pad

        #转换成(m, h_, w_, c*kh*kw)
        mat = mat.reshape(m, h_, w_, c*kh*kw)

        #还原成填充后的特征图
        padded = np.zeros((m, c, h + 2*ph, w + 2*pw))
        for i in range(h_):
            for j in range(w_):
                padded[:, :, i*kh:(i+1)*kh, j*kw:(1+j)kw] += mat[:, i, j, :].reshape(m, c, kh, kw)

        #得到原图(m,c,h,w)
        out = padded[:, :, h+ph, w+pw]

        return out

    '''
    向前传播
    in_batch: 一批输入数据
    training: 是否正在训练
    '''
    def forward(self, in_batch, training=False):
        W = self.__W.value
        b = self.__b.value
        self.__in_batch_shape = in_batch.shape

        #把输入特征图展开成卷积运算的矩阵矩阵(m*h_*w_, c*kh*kw)
        in_batch = self.__img2mat(in_batch)
        #计算输出值(m*h_*w_, c_) = (m*h_*w_, c*kh*kw) @ (c*kh*kw, c_) + (c_,)
        out = in_batch @ W + b
        #把输出值还原成(m, c_, h_, w_)
        out = np.moveaxis(out, 1, -1)

        self.__in_batch = in_batch

        if self.__outshape != self.__real_outshape:
            out = out.reshape(self.__outshape)

        return self.activation(out)

    #反向传播梯度
    def backward(self, gradient):
        if self.__outshape != self.__real_outshape:
            gradient = gradient.reshape(self.__real_outshape)

        W = self.__W.value

        #(m, c_, h_, w_)
        grad = self.activation.grad(gradient)

        #把形状转换成(m, h_, w_, c_)
        grad = np.moveaxis(grad, -1, 1)
        #把形状转换成(m*h_*w_, c_)
        m, h_, w_, c_ = grad.shape
        grad = grad.reshape((m*h_*w_, c_))

        #计算W参数梯度(c*kh*kw, c_) = (c*kh*kw, m*h_*w_)  @ (m*h_*w_, c_)
        dW = self.__in_batch.T @ grad
        #计算b参数梯度(c_,)
        db = grad.sum(axis=0)

        self.__W.gradient = dW
        self.__b.gradient = db

        #计算in_batch梯度 (m*h_*w_, c*kh*kw) = (m*h_*w_, c_) @ (c_, c*kh*kw)
        dIn = grad @ W.T

        #把梯度矩阵转换成输入特征图的形状(m,c,h,w)
        dIn = self.__mat2img(dIn)

        return dIn

    #重置当前层的状态
    def reset(self):
        self.__in_batch = None
        self.__in_batch_shape = None

        self.__W = LayerParam.reset(self.__W)
        self.__b = LayerParam.reset(self.__b)
