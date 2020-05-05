# coding=utf-8

'''
卷积层实现
'''
import pdb
import copy
import numpy as np
from model import Layer, LayerParam
import utils

'''
计算2D卷积层的输输出和填充
'''
def compute_2D_outshape(inshape, kernel_size, strides, padding):
    kh, kw = kernel_size
    sh, sw = strides

    h_ = -1
    w_ = -1
    pad = (0, 0)
    if 'same' == padding:
        _, h_, w_ = inshape
        pad = ((h_*sh - h + kh - 1)//2, (w_*sw - w + kw - 1)//2)
    elif 'valid' == padding:
        _, h, w = inshape
        h_ = (h - kh + 1)//sh
        w_ = (w - kw + 1)//sw
    else:
        raise Exception("invalid padding: "+padding)

    #pdb.set_trace()
    outshape = (h_, w_)
    return outshape, pad

'''
把2D特征图转换成方便卷积运算的矩阵
'''
def img2D_mat(img, kernel_size, pad, strides):
    kh, kw = kernel_size
    ph, pw = pad
    sh, sw = strides
    #pdb.set_trace()
    m, c, h, w = img.shape
    kh,kw = self.__ks

    #先填充
    padded = np.zeros((m, c, h + 2*ph, w + 2*pw))
    padded[:, :, ph:(ph+h), pw:(pw+w)] = img

    h_ = (padded.shape[2] - kh + 1)//sh
    w_ = (padded.shape[3] - kw + 1)//sw

    #转换成矩阵(m, h_, w_, c*kh*kw)
    #pdb.set_trace()
    out = np.zeros((m, h_, w_, c*kh*kw))
    for i in range(h_):
        for j in range(w_):
            #(m, c, kh, kw)
            cov = padded[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
            #(m, c*kh*kw)
            cov = cov.reshape((m, c*kh*kw))
            out[:, i, j, :] = cov

    #转换成(m*h_*w_, c*kh*kw)
    out = out.reshape((-1, c*kh*kw))

    return out

'''
卷积运算矩阵转换成2D特征图
'''
def mat_img2D(mat, imgshape, kernel_size, pad, strides):
    m, c, h, w = imgshape
    kh, kw = kernel_size
    sh, sw = strides
    ph, pw = pad

    padded = np.zeros((m, c, h + 2*ph, w + 2*pw))
    h_ = (padded[2] - kh + 1)//sh
    w_ = (padded[3] - kw + 1)//sw

    #转换(m*h_*w_, c*kh*kw)->(m, h_, w_, c*kh*kw)
    mat = mat.reshape(m, h_, w_, c*kh*kw)
    #还原成填充后的特征图
    for i in range(h_):
        for j in range(w_):
            padded[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += mat[:, i, j, :].reshape(m, c, kh, kw)

    #pdb.set_trace()
    #得到原图(m,c,h,w)
    out = padded[:, :, ph:h+ph, pw:w+pw]

    return out


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
class Conv2D(Layer):
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
    def __init__(self,
                channels,
                kernel_size,
                strides=(1,1),
                padding='same',
                inshape=None,
                activation='relu'):
        #pdb.set_trace()
        self.__ks = kernel_size
        self.__st = strides
        self.__pad = (0, 0)
        self.__padding = padding

        #参数
        self.__W = None #(c*kernel_size, c_)
        self.__b = None #(c_)

        #输入输出形状
        self.__inshape = (-1, -1, -1)
        self.__outshape = None

        #输出形状
        outshape = self.check_shape(channels)
        if outshape is None or type(channels) != type(1):
            raise Exception("invalid channels: "+str(channels))

        self.__outshape = outshape

        #输入形状
        inshape = self.check_shape(inshape)
        if self.valid_shape(inshape):
            self.__inshape = self.check_shape(inshape)
            if self.__inshape is None or len(self.__inshape) != 3:
                raise Exception("invalid inshape: "+str(inshape))

            outshape, self.__pd = compute_2D_outshape(self.__inshape, self.__ks, self.__st, self.__padding)
            self.__outshape = self.__outshape + outshape

        super().__init__(activation)

        self.__in_batch_shape = None
        self.__in_batch = None

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    @property
    def params(self):
        return [self.__W, self.__b]

    def init_params(self):
        inshape = self.__inshape
        outshape = self.__outshape
        in_chnls = inshape[0]
        out_chnls = outshape[0]

        #pdb.set_trace()
        #展平形状(c*kh*kw, c_), 把卷积运算转换成矩阵运算, 优化性能
        shape = (in_chnls * utils.flat_shape(k), out_chnls)
        std = 0.01
        wval = np.random.randn(shape[0], shape[1]) * std
        bval = np.zeros(shape[1])

        W = LayerParam(self.name, 'weight', wval)
        b = LayerParam(self.name, 'bias', bval)

        self.__W = W
        self.__b = b

    def set_prev(self, prev_layer):
        inshape = prev_layer.outshape
        self.__inshape = inshape
        outshape, self.__pd = compute_2D_outshape(inshape, self.__ks, self.__st, self.__padding)
        self.__outshape = self.__outshape + outshape

        super().set_prev(prev_layer)

    '''
    向前传播
    in_batch: 一批输入数据
    training: 是否正在训练
    '''
    def forward(self, in_batch, training=False):
        #pdb.set_trace()
        W = self.__W.value
        b = self.__b.value
        self.__in_batch_shape = in_batch.shape

        #把输入特征图展开成卷积运算的矩阵矩阵(m*h_*w_, c*kh*kw)
        in_batch = img2D_mat(in_batch, self.__ks, self.__pad, self.__st)
        #计算输出值(m*h_*w_, c_) = (m*h_*w_, c*kh*kw) @ (c*kh*kw, c_) + (c_,)
        out = in_batch @ W + b
        #把(m*h_*w_, c_) 转换成(m, h_, w_, c_)
        c_, h_, w_ = self.__outshape
        out = out.reshape((-1, h_, w_, c_))
        #把输出值还原成(m, c_, h_, w_)
        out = np.moveaxis(out, -1, 1)

        self.__in_batch = in_batch

        return self.activation(out)

    #反向传播梯度
    def backward(self, gradient):
        #pdb.set_trace()
        W = self.__W.value

        #(m, c_, h_, w_)
        grad = self.activation.grad(gradient)
        #把形状转换成(m, h_, w_, c_)
        grad = np.moveaxis(grad, 1, -1)
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
        dIn = mat_img2D(dIn, self.__in_batch_shape, self.__ks, self.__pad, self.__strides)

        return dIn

    #重置当前层的状态
    def reset(self):
        self.__in_batch = None
        self.__in_batch_shape = None

        self.__W = LayerParam.reset(self.__W)
        self.__b = LayerParam.reset(self.__b)


'''
2D最大池化层
'''
class MaxPool2D(Layer):

    def __init__(self, pool_size=(2,2), strides=(1,1), padding='valid'):
        self.__ks = pool_size
        self.__st = strides
        self.__padding = padding
        self.__pad = None

        self.__inshape = (-1,-1,-1)
        self.__outshape = (-1,-1,-1)

        self.__mark = None

        super().__init__()


    def init_params(self):
        pass

    @property
    def params(self):
        return []

    @property
    def inshape(self):
        return self.__inshape

    @property
    def outshape(self):
        return self.__outshape

    def set_prev(self, prev_layer):
        inshape = prev_layer.outshape
        self.__inshape = inshape
        outshape, self.__pad = compute_2D_outshape(inshape, self.__ks, self.__st, self.__padding)
        self.__outshape = (self.__inshape[0],)+outshape

        super().set_prev(prev_layer)

    def forward(self, in_batch, training=False):
        m, c, h, w = in_batch.shape
        _, h_, w_ = self.outshape
        kh, kw = self.__ks
        #把特征图转换成矩阵(m, c, h, w)->(m*h_*w_, c*kh*kw)
        in_batch = img2D_mat(in_batch, self.outshape, self.__ks, self.__pd, self.__st)
        #转换形状(m*w_*h_, c*kh*kw)->(m*h_*w_, c, kh*kw)
        in_batch = in_batch.reshape((m*h_*w_, c, kh*kw))
        #得到最大值(m*h_*w_, c, 1)
        max = np.max(in_batch, axis=2).reshape(m*h_*w_, c, 1)
        #得到最大值的索引(m*h_*w_,c, kh*kw)
        self.__mark = in_batch == max

        #得到输出值
        out = out.reshape(m, h_, w_, c)
        out = np.moveaxis(out, -1, 1)

        return out

    def backward(self, gradient):
        c, h_, w_ = self.outshape
        m = gradient.shape[0]
        kh, kw = self.__ks
        #(m, c, h_, w_) -> (m, h_, w_, c)
        grad = gradient.moveaxis(1, -1)
        #(m, h_, w_, c) -> (m*h_*w_, c)
        grad = grad.reshape((m*h_*w_, c))
        #还原矩阵(m*h_*w_, c, kh*kw) -> (m*h_*w_, c*kh*kw)
        mat = np.zeros((m*h_*w_, c, kh*kw))
        mat[self.__mark] = grad.reshape(-1)
        #把矩阵还原成特征图
        out = mat_img2D(mat, self.inshape, self.__ks, self.__pd, self.__st)

        return out

    def reset(self):
        self.__mark = None
