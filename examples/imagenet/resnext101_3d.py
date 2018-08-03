# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.ConvolutionND(3,
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.ConvolutionND(3,
                ch, ch, 3, 1, 1, groups=32, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.ConvolutionND(3,
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.ConvolutionND(3,
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.ConvolutionND(3,
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.ConvolutionND(3,
                ch, ch, 3, 1, 1, groups=32, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.ConvolutionND(3,
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block3D(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block3D, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNeXt101(chainer.Chain):

    insize = 224

    def __init__(self):
        super(ResNeXt101, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(3,
                3, 64, 7, stride=(1, 2, 2), pad=(3, 3, 3), initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block3D(3, 64, 128, 256, 1)
            self.res3 = Block3D(24, 256, 256, 512)
            self.res4 = Block3D(36, 512, 512, 1024)
            self.res5 = Block3D(3, 1024, 1024, 2048)
            self.fc = L.Linear(2048, 1000)

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_nd(F.relu(h), 3, stride=2, pad=1)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        #h = F.average_pooling_nd(h, 7, stride=1)
        h = F.average(h, axis=(2, 3, 4), keepdims=True) #global average pooling
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
