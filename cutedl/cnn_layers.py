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
    #pdb.set_trace()
    _, h, w = inshape
    kh, kw = kernel_size
    sh, sw = strides

    h_ = -1
    w_ = -1
    pad = (0, 0)
    if 'same' == padding:
        #填充, 使用输入输出形状一致
        _, h_, w_ = inshape
        pad = (((h_-1)*sh - h + kh )//2, ((w_-1)*sw - w + kw)//2)
    elif 'valid' == padding:
        #不填充
        h_ = (h - kh)//sh + 1
        w_ = (w - kw)//sw + 1
    else:
        raise Exception("invalid padding: "+padding)

    #pdb.set_trace()
    outshape = (h_, w_)
    return outshape, pad

'''
把2D特征图转换成方便卷积运算的矩阵, 形状(m*h_*w_, c*kh*kw)
img 特征图 shape=(m,c,h,w)
kernel_size 核形状 shape=(kh, kw)
pad 填充大小 shape=(ph, pw)
strides 步长 shape=(sh, sw)
'''
def img2D_mat(img, kernel_size, pad, strides):
    #pdb.set_trace()
    kh, kw = kernel_size
    ph, pw = pad
    sh, sw = strides
    #pdb.set_trace()
    m, c, h, w = img.shape
    kh, kw = kernel_size

    #得到填充的图
    pdshape = (m, c) + (h + 2*ph, w + 2*pw)
    #得到输出大小
    h_ = (pdshape[2] - kh)//sh + 1
    w_ = (pdshape[3] - kw)//sw + 1
    #填充
    padded = np.zeros(pdshape)
    padded[:, :, ph:(ph+h), pw:(pw+w)] = img

    #转换成卷积矩阵(m, h_, w_, c, kh, kw)
    #pdb.set_trace()
    out = np.zeros((m, h_, w_, c, kh, kw))
    for i in range(h_):
        for j in range(w_):
            #(m, c, kh, kw)
            cov = padded[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
            out[:, i, j] = cov

    #转换成(m*h_*w_, c*kh*kw)
    out = out.reshape((-1, c*kh*kw))

    return out

'''
矩阵形状的梯度转换成2D特征图梯度
mat 矩阵梯度 shape=(m*h_*w_, c*kh*kw)
特征图形状 imgshape=(m, c, h, w)
'''
def matgrad_img2D(mat, imgshape, kernel_size, pad, strides):
    #pdb.set_trace()
    m, c, h, w = imgshape
    kh, kw = kernel_size
    sh, sw = strides
    ph, pw = pad

    #得到填充形状
    pdshape = (m, c) + (h + 2*ph, w + 2 * pw)
    #得到输出大小
    h_ = (pdshape[2] - kh)//sh + 1
    w_ = (pdshape[3] - kw)//sw + 1

    #转换(m*h_*w_, c*kh*kw)->(m, h_, w_, c, kh, kw)
    mat = mat.reshape(m, h_, w_, c, kh, kw)

    #还原成填充后的特征图
    padded = np.zeros(pdshape)
    for i in range(h_):
        for j in range(w_):
            #(m, c, kh, kw)
            padded[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += mat[:, i, j]

    #pdb.set_trace()
    #得到原图(m,c,h,w)
    out = padded[:, :, ph:ph+h, pw:pw+w]

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

h_ = (h + 2*ph - kh)//sh + 1
w_ = (w + 2*pw - kw)//sw + 1
不填充 ph=pw=0
填充到原来的大小: h_=h w_=w
ph = (sh*(h-1) - h + kh)/2, 当sh=1时 ph = (kh-sh)/2
pw = (sw*(w-1) - w + kw)/2, 当sw=1时 pw = (kw-sh)/2
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
    kernel_initialier 卷积核初始化器
            uniform 均匀分布
            normal  正态分布
    bias_initialier 偏移量初始化器
            uniform 均匀分布
            normal 正态分布
            zeros  0
    '''
    def __init__(self, channels, kernel_size, strides=(1,1),
                padding='same',
                inshape=None,
                activation='relu',
                kernel_initializer='uniform',
                bias_initializer='zeros'):
        #pdb.set_trace()
        self.__ks = kernel_size
        self.__st = strides
        self.__pad = (0, 0)
        self.__padding = padding

        #参数
        self.__W = self.weight_initializers[kernel_initializer]
        self.__b = self.bias_initializers[bias_initializer]

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

            outshape, self.__pad = compute_2D_outshape(self.__inshape, self.__ks, self.__st, self.__padding)
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
        shape = (in_chnls * utils.flat_shape(self.__ks), out_chnls)
        wval = self.__W(shape)
        bval = self.__b((shape[1], ))

        W = LayerParam(self.name, 'weight', wval)
        b = LayerParam(self.name, 'bias', bval)

        self.__W = W
        self.__b = b

    def set_prev(self, prev_layer):
        #pdb.set_trace()
        inshape = prev_layer.outshape
        self.__inshape = inshape
        outshape, self.__pad = compute_2D_outshape(inshape, self.__ks, self.__st, self.__padding)
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
        #把形状转换成(m, c_, h_, w_) -> (m, h_, w_, c_)
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
        dIn = matgrad_img2D(dIn, self.__in_batch_shape, self.__ks, self.__pad, self.__st)

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

    def __init__(self, pool_size=(2,2), strides=(2,2), padding='valid'):
        self.__ks = pool_size
        self.__st = strides
        self.__padding = padding
        self.__pad = (0,0)

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
        #pdb.set_trace()
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
        in_batch = img2D_mat(in_batch, self.__ks, self.__pad, self.__st)
        #转换形状(m*w_*h_, c*kh*kw)->(m*h_*w_*c,kh*kw)
        in_batch = in_batch.reshape((m*h_*w_*c, kh*kw))
        #得到最大最索引
        idx = in_batch.argmax(axis=1).reshape(-1, 1)
        #转成in_batch相同的形状
        idx = idx @ np.ones((1, in_batch.shape[1]))
        temp = np.ones((in_batch.shape[0], 1)) @ np.arange(in_batch.shape[1]).reshape(1, -1)
        #得到boolean的标记
        self.__mark = idx == temp

        #得到最大值
        max = in_batch[self.__mark]
        max = max.reshape((m, h_, w_, c))
        max = np.moveaxis(max, -1, 1)

        return max

    def backward(self, gradient):
        c, h_, w_ = self.outshape
        m = gradient.shape[0]
        kh, kw = self.__ks
        #(m, c, h_, w_) -> (m, h_, w_, c)
        grad = np.moveaxis(gradient, 1, -1)
        #(m*h_*w_*c, kh*kw)
        mat = np.zeros((m*h_*w_*c, kh*kw))
        mat[self.__mark] = grad.reshape(-1)
        mat = mat.reshape((m*h_*w_, c*kh*kw))
        #pdb.set_trace()
        #把矩阵还原成特征图
        out = matgrad_img2D(mat, (m,)+self.inshape, self.__ks, self.__pad, self.__st)

        return out

    def reset(self):
        self.__mark = None
