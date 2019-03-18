#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

import random

class Block3D(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, pad=1):
        super(Block3D, self).__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=out_channels, ksize=ksize, pad=pad)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)


class C3D_lstm(chainer.Chain):

    """
    C3D_lstm
    - It takes (112, 112, 16) sized image as imput
    """
    insize = 112

    def __init__(self):
        super(C3D_lstm, self).__init__()
        with self.init_scope():
            self.conv1a = Block3D(3, 64, 3)
            self.conv2a = Block3D(64, 128, 3)
            self.conv3a = Block3D(128, 256, 3)
            self.conv3b = Block3D(256, 256, 3)
            self.conv4a = Block3D(256, 512, 3)
            self.conv4b = Block3D(512, 512, 3)
            self.conv5a = Block3D(512, 512, 3)
            self.conv5b = Block3D(512, 512, 3)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

            self.lstm1 = L.LSTM(1000, 1000, lateral_init=chainer.initializers.Normal(scale=0.01))
            #self.lstm2 = L.LSTM(1000, 1000, lateral_init=chainer.initializers.Normal(scale=0.01))
            #self.lstm3 = L.LSTM(1000, 1000, lateral_init=chainer.initializers.Normal(scale=0.01))

            self.count = 0

    def reset_state(self):
        self.lstm1.reset_state() 
    
    def __call__(self, x, t):

        self.reset_state() 

        aaa = 64
        current_length = aaa-(self.count) % aaa
        
        #b = [64, 64, 64, 56, 48, 40, 32, 24, 16, 8]
        b = [16, 8, 4]
        current_length = b[(self.count) % len(b)]

        self.count += 1
        print("current_length: ", current_length)
        chainer.advise_num_inputs(current_length)

        loss = 0
        for i in range(current_length):
            #print(i)
            h = self.conv1a(x)
            h = F.max_pooling_nd(h, ksize=(1, 2, 2), stride=(1, 2, 2))
            h = self.conv2a(h)
            h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
            h = self.conv3a(h)
            h = self.conv3b(h)
            h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
            h = self.conv4a(h)
            h = self.conv4b(h)
            h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
            h = self.conv5a(h)
            h = self.conv5b(h)
            h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
            h = self.fc6(h)
            h = F.relu(h)
            h = F.dropout(h, ratio=0.5)
            h = self.fc7(h)
            h = F.relu(h)
            h = F.dropout(h, ratio=0.5)
            h = self.fc8(h)
            h = self.lstm1(h)
            #h = self.lstm2(h)
            #h = self.lstm3(h)

            loss += F.softmax_cross_entropy(h, t)
        
        #chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.acc = F.accuracy(h, t)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred
