#!/usr/bin/env python

from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.dataset import convert
from chainer.training import extensions
from chainer.training import updaters

import cleanup
aaa = [448, 448, 392, 392, 338, 338, 280]

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, mean, crop_size, random=True):
        #self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return 256

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        
        #crop_height_size = self.crop_size #(i//8+1)*8  #self.crop_size + (i%10)*8 
        #crop_height_size = self.crop_size + (i%10)*8 
        #crop_width_size = self.crop_size

        
        crop_height_size = 1664 + 8*i
        crop_width_size = 448*6 + 8*128

        input_image = np.random.rand(1, crop_height_size, crop_width_size)
        test_image = np.random.rand(1, crop_height_size, crop_width_size)
        
        input_image = input_image.astype(np.float32)
        test_image = test_image.astype(np.float32)

        return input_image, test_image


def concat_examples_with_padding(batch, device=None):
    return convert.concat_examples(batch, device, padding=0)

def main():
    archs = {
        'cleanup': cleanup.CleanupNet
    }
    optimize_settings = ['keep_all', 'swap_all_no_scheduling', 'swap_all', 'recompute_all', 'swap_opt', 'superneurons']


    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    #parser.add_argument('train', help='Path to training image-label list file')
    #parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='cleanup', help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--iteration', '-i', type=int, default=0,
                        help='Number of iterations to train')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",
                        default=[0, 1, 2, 3])
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
    parser.add_argument('--optimize_setting', choices=optimize_settings,
                        default=None, help='ooc optimize setting')
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

    model.converter = concat_examples_with_padding

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(mean, model.insize)
    val = PreprocessedDataset(mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.

    devices = tuple(args.gpus)

    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               args.batchsize,
                                               n_processes=args.loaderjob)
        for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
#    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer = chainer.optimizers.MomentumSGD(lr=0.16, momentum=0.9)
#    optimizer = chainer.optimizers.MomentumSGD(lr=0.32, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Set up a trainer
    updater = updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                   devices=devices)
    if args.iteration > 0:
        trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)
    else:
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 1, 'epoch'
        log_interval = 1000, 'iteration'
#        val_interval = 100000, 'iteration'
#        log_interval = 1000, 'iteration'

    lr_interval = (1 if args.test else 30), 'epoch'
    snapshot_interval = (1 if args.test else 1), 'epoch'

    #trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=lr_interval)
    #trainer.extend(extensions.Evaluator(val_iter, model, converter=concat_examples_with_padding, device=args.gpus[0]),
    #               trigger=val_interval)
    #trainer.extend(extensions.dump_graph('main/loss'))
    #trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot(), trigger=val_interval)
    #trainer.extend(extensions.snapshot_object(
    #    model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    log_interval = (1, 'epoch')
    trainer.extend(extensions.LogReport(trigger=log_interval))
    #trainer.extend(extensions.observe_lr(), trigger=log_interval)
    #trainer.extend(extensions.PrintReport([
    #    'epoch', 'iteration', 'main/loss', 'validation/main/loss',
    #    'main/accuracy', 'validation/main/accuracy', 'lr'
    #]), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'elapsed_time']), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration']), trigger=(10, 'iteration'))
    #trainer.extend(extensions.ProgressBar(update_interval=2))

    model.compute_accuracy = False

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.ooc:
        with chainer.out_of_core_mode(fine_granularity=True, devices=devices, optimize_setting=args.optimize_setting):
            trainer.run()
    else:
            trainer.run()

if __name__ == '__main__':
    main()
