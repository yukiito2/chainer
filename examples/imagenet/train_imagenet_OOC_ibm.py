#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import alex
import googlenet
import googlenetbn
import nin
import resnet50


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, mean, crop_size, random=True):
        #self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return 5000

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image = np.random.rand(3, crop_size, crop_size)
        label = np.array(1, np.int32)
        
        image = image.astype(np.float32)
        label *= random.randint(0, 99)

        return image, label


def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    #parser.add_argument('train', help='Path to training image-label list file')
    #parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--iteration', '-i', type=int, default=0,
                        help='Number of iterations to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--ooc',
                    action='store_true', default=False,
                    help='Functions of out-of-core')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--debug', '-d', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    # Check cudnn version
    import cupy
    if chainer.cuda.cudnn_enabled:
        cudnn_v = cupy.cudnn._cudnn_version
        print('cuDNN Version:', cudnn_v)

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(mean, model.insize)
    val = PreprocessedDataset(mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.04, momentum=0.9)
    # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    if args.iteration > 0:
        trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)
    else:
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 1), 'epoch'
    #val_interval = (10 if args.test else 100000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'
    #log_interval = (10 if args.test else 10), 'iteration'

    lr_interval = (1 if args.test else 30), 'epoch'
    snapshot_interval = (1 if args.test else 1), 'epoch'
    #snapshot_interval = (1 if args.test else 40000), 'iteration'

    trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=lr_interval)
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    #trainer.extend(extensions.dump_graph('main/loss'))
    #trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot(), trigger=val_interval)
    #trainer.extend(extensions.snapshot_object(
    #    model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    #trainer.extend(extensions.LogReport(trigger=log_interval))
    #trainer.extend(extensions.observe_lr(), trigger=log_interval)
    #trainer.extend(extensions.PrintReport([
    #    'epoch', 'iteration', 'main/loss', 'validation/main/loss',
    #    'main/accuracy', 'validation/main/accuracy', 'lr'
    #]), trigger=log_interval)
    #trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.ooc:
        with chainer.out_of_core_mode(fine_granularity=True):
            trainer.run()
    else:
        trainer.run()

if __name__ == '__main__':
    main()