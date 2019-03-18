#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

class CleanupNet(chainer.Chain):

    """
    CleanupNet
    """
    insize = 448

    def __init__(self):
        super(CleanupNet, self).__init__(
            down_conv1=L.Convolution2D(1, 48, 5, stride=2, pad=2),
            flat_conv1_1=L.Convolution2D(48, 128, 3, stride=1, pad=1),
            flat_conv1_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            down_conv2=L.Convolution2D(128, 256, 3, stride=2, pad=1),
            flat_conv2_1=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            flat_conv2_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            down_conv3=L.Convolution2D(256, 256, 3, stride=2, pad=1),
            flat_conv3_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            flat_conv3_2=L.Convolution2D(512, 1024, 3, stride=1, pad=1),
            flat_conv3_3=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            flat_conv3_4=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            flat_conv3_5=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),
            flat_conv3_6=L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            flat_conv3_7=L.Convolution2D(512, 256, 3, stride=1, pad=1),

            up_conv4=L.Deconvolution2D(256, 256, 4, stride=2, pad=1),
            flat_conv4_1=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            flat_conv4_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            up_conv5=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            flat_conv5_1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            flat_conv5_2=L.Convolution2D(128, 48, 3, stride=1, pad=1),

            up_conv6=L.Deconvolution2D(48, 48, 4, stride=2, pad=1),
            flat_conv6_1=L.Convolution2D(48, 24, 3, stride=1, pad=1),
            flat_conv6_2=L.Convolution2D(24, 1, 3, stride=1, pad=1),

            #bn1_1 = L.BatchNormalization(128),
            #bn1_2 = L.BatchNormalization(128),
            #bn2_1 = L.BatchNormalization(256),
            #bn2_2 = L.BatchNormalization(256),
            #bn3_1 = L.BatchNormalization(512),
            #bn3_2 = L.BatchNormalization(1024),
            #bn3_3 = L.BatchNormalization(1024),
            #bn3_4 = L.BatchNormalization(1024),
            #bn3_5 = L.BatchNormalization(1024),
            #bn3_6 = L.BatchNormalization(512),
            #bn3_7 = L.BatchNormalization(256),
            #bn4_1 = L.BatchNormalization(256),
            #bn4_2 = L.BatchNormalization(256),
            #bn5_1 = L.BatchNormalization(128),
            #bn5_2 = L.BatchNormalization(48),
            #bn6_1 = L.BatchNormalization(24),
            #bn6_2 = L.BatchNormalization(1)
        )
        self.train = False

    def __call__(self, x, t):
        
        #chainer.advise_num_inputs(1)
        
        h = F.relu(self.down_conv1(x))
        h = F.relu(self.flat_conv1_1(h))
        #h = self.bn1_1(h)
        h = F.relu(self.flat_conv1_2(h))
        #h = self.bn1_2(h)

        h = F.relu(self.down_conv2(h))
        h = F.relu(self.flat_conv2_1(h))
        #h = self.bn2_1(h)
        h = F.relu(self.flat_conv2_2(h))
        #h = self.bn2_2(h)

        h = F.relu(self.down_conv3(h))
        h = F.relu(self.flat_conv3_1(h))
        #h = self.bn3_1(h)
        h = F.relu(self.flat_conv3_2(h))
        #h = self.bn3_2(h)
        h = F.relu(self.flat_conv3_3(h))
        #h = self.bn3_3(h)
        h = F.relu(self.flat_conv3_4(h))
        #h = self.bn3_4(h)
        h = F.relu(self.flat_conv3_5(h))
        #h = self.bn3_5(h)
        h = F.relu(self.flat_conv3_6(h))
        #h = self.bn3_6(h)
        h = F.relu(self.flat_conv3_7(h))
        #h = self.bn3_7(h)

        h = F.relu(self.up_conv4(h))
        h = F.relu(self.flat_conv4_1(h))
        #h = self.bn4_1(h)
        h = F.relu(self.flat_conv4_2(h))
        #h = self.bn4_2(h)

        h = F.relu(self.up_conv5(h))
        h = F.relu(self.flat_conv5_1(h))
        #h = self.bn5_1(h)
        h = F.relu(self.flat_conv5_2(h))
        #h = self.bn5_2(h)

        h = F.relu(self.up_conv6(h))
        h = F.relu(self.flat_conv6_1(h))
        #h = self.bn6_1(h)
        h = F.relu(self.flat_conv6_2(h))
        #h = self.bn6_2(h)

        h = F.sigmoid(h)

        loss = F.mean_squared_error(h, t)
        chainer.report({'loss': loss}, self)
        return loss
