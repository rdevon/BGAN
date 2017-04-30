#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Binary MNIST with BGAN.

'''

from __future__ import print_function

import argparse
import cPickle
import datetime
import gzip
import logging
import os
from os import path
import sys
import time

from collections import OrderedDict
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
import lasagne
from lasagne.layers import (
    batch_norm, DenseLayer, GaussianNoiseLayer, InputLayer, ReshapeLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid, softmax, tanh
from matplotlib import pylab as plt
import numpy as np
from PIL import Image
from progressbar import Bar, ProgressBar, Percentage, Timer
import pylab as pl
import random
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc


floatX = lasagne.utils.floatX
floatX_ = theano.config.floatX
lrelu = LeakyRectify(0.2)

DIM_X = 28
DIM_Y = 28
DIM_C = 1

# ##################### UTIL #####################

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.propagate = False
file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]:%(message)s')
stream_formatter = logging.Formatter(
    '[%(levelname)s:%(name)s]:%(message)s' + ' ' * 40)

def set_stream_logger(verbosity):
    global logger

    if verbosity == 0:
        level = logging.WARNING
        lstr = 'WARNING'
    elif verbosity == 1:
        level = logging.INFO
        lstr = 'INFO'
    elif verbosity == 2:
        level = logging.DEBUG
        lstr = 'DEBUG'
    else:
        level = logging.INFO
        lstr = 'INFO'
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.terminator = ''
    ch.setLevel(level)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    logger.info('Setting logging to %s' % lstr)

def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    fh.terminator = ''
    logger.info('Saving logs to %s' % file_path)
    
def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]

# ############################# DATA ###############################

def load_dataset(source, mode):
    if source is None:
        raise ValueError('source not provided.')
    
    logger.info('Reading MNIST ({}), from {}'.format(mode, source))
    with gzip.open(source, 'rb') as f:
        x = cPickle.load(f)

    if mode == 'train':
        data = np.float32(x[0][0])
    elif mode == 'valid':
        data = np.float32(x[1][0])
    elif mode == 'test':
        data = np.float32(x[2][0])
    else:
        raise ValueError()

    data = np.reshape(data, (-1, 1, 28, 28))
    return data

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

# ##################### MODEL #######################

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0, W=None, b=None,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        if W is None:
            self.W = self.add_param(lasagne.init.Orthogonal(),
                                    (self.input_shape[1], num_filters) + self.filter_size,
                                    name='W')
        else:
            self.W = self.add_param(W,
                                    (self.input_shape[1], num_filters) + self.filter_size,
                                    name='W')
        if b is None:
            self.b = self.add_param(lasagne.init.Constant(0),
                                    (num_filters,),
                                    name='b')
        else:
            self.b = self.add_param(b,
                                    (num_filters,),
                                    name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i * s - 2 * p + f - 1
                      for i, s, p, f in zip(input_shape[2:],
                                            self.stride,
                                            self.pad,
                                            self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def build_generator(input_var=None, dim_z=100, dim_h=64):
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, dim_h * 2 * 7 * 7))
    layer = ReshapeLayer(layer, ([0], dim_h * 2, 7, 7))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2,
                          nonlinearity=None)
    
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def build_discriminator(input_var=None, dim_h=64):
    layer = InputLayer(shape=(None, 1, DIM_X, DIM_Y), input_var=input_var)
    
    layer = Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2,
                        nonlinearity=lrelu)
    layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=None)
    
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer

# ##################### MATH #######################

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def norm_exp(log_factor):
    '''Gets normalized weights.

    '''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX_)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

# ##################### LOSS #######################

def BGAN(discriminator, g_output_logit, n_samples, trng, batch_size=64):
    d = OrderedDict()
    R = trng.uniform(size=(n_samples, batch_size, DIM_C, DIM_X, DIM_Y),
                     dtype=floatX_)
    g_output = T.nnet.sigmoid(g_output_logit)
    samples = (R <= T.shape_padleft(g_output)).astype(floatX_)

    # Create expression for passing real data through the discriminator
    D_r = lasagne.layers.get_output(discriminator)
    D_f = lasagne.layers.get_output(
        discriminator, samples.reshape((-1, DIM_C, DIM_X, DIM_Y)))
    D_f_ = D_f.reshape((n_samples, batch_size))
    
    log_d1 = -T.nnet.softplus(-D_f_)
    log_d0 = -(D_f_ + T.nnet.softplus(-D_f_))
    log_w = log_d1 - log_d0
    log_g = -((1. - samples) * T.shape_padleft(g_output_logit)
             + T.shape_padleft(T.nnet.softplus(-g_output_logit))).sum(
        axis=(2, 3, 4))

    log_N = T.log(log_w.shape[0]).astype(floatX_)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)

    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-D_r)).mean() + (
        T.nnet.softplus(-D_f)).mean() + D_f.mean()
    
    return generator_loss, discriminator_loss, D_r, D_f, log_Z_est, log_w, w_tilde, d

# ############################## MAIN ################################

def summarize(results, samples, image_dir=None, prefix=''):
    results = dict((k, np.mean(v)) for k, v in results.items())
    logger.info(results)
    if image_dir is not None:
        plt.imsave(path.join(image_dir, '{}.png'.format(prefix)),
                   (samples.reshape(10, 10, 28, 28)
                    .transpose(0, 2, 1, 3)
                    .reshape(10 * 28, 10 * 28)),
                   cmap='gray')

def main(source=None, num_epochs=None, method=None, batch_size=None,
         learning_rate=None, beta=None, n_samples=None,
         image_dir=None, binary_dir=None,
         dim_z=None, prior=None):
    
    # DATA
    data = load_dataset(source=source, mode='train')
    train_samples = data.shape[0]

    # VAR
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    log_Z = theano.shared(floatX(0.), name='log_Z')
    
    # MODEL
    logger.info('Building model and graph')
    generator = build_generator(noise_var, dim_z=dim_z)
    discriminator = build_discriminator(input_var)
    
    # RNG
    trng = RandomStreams(random.randint(1, 1000000))
    
    # GRAPH / LOSS
    g_output_logit = lasagne.layers.get_output(generator)
    generator_loss, discriminator_loss, D_r, D_f, log_Z_est, log_w, w_tilde, d = BGAN(
        discriminator, g_output_logit, n_samples, trng)

    # OPTIMIZER
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator,
                                                         trainable=True)
    
    eta = theano.shared(floatX(learning_rate))
    
    updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=eta, beta1=beta)
    updates.update(lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=eta, beta1=beta))
    updates.update([(log_Z, 0.95 * log_Z + 0.05 * log_Z_est.mean())])

    # COMPILE
    results = {
        'p(real)': (T.nnet.sigmoid(D_r) > .5).mean(),
        'p(fake': (T.nnet.sigmoid(D_f) < .5).mean(),
        'G loss': generator_loss,
        'D loss': discriminator_loss,
        'log Z': log_Z,
        'log Z est': log_Z_est.mean(),
        'log_Z est var': log_Z_est.std() ** 2,
        'log w': log_w.mean(),
        'log w var': log_w.std() ** 2,
        'norm w': w_tilde.mean(),
        'norm w var': w_tilde.std() ** 2,
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean()
    }
    train_fn = theano.function([noise_var, input_var],
                               results,
                               updates=updates)

    gen_fn = theano.function(
        [noise_var], T.nnet.sigmoid(lasagne.layers.get_output(
            generator, deterministic=True)))

# TRAIN
    logger.info('Training...')
    
    results = {}
    for epoch in range(num_epochs):
        u = 0
        prefix = '{}_{}'.format(method, epoch)
        
        e_results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        prefix = str(epoch)
        
        start_time = time.time()
        batch0 = None
        for batch in iterate_minibatches(data, batch_size, shuffle=True):
            if batch0 is None: batch0 = batch
            if batch.shape[0] == batch_size:
                if prior == 'uniform':
                    noise = floatX(np.random.rand(batch_size, dim_z))
                elif prior == 'gaussian':
                    noise = floatX(np.random.normal(size=(batch_size, dim_z)))
                    
                outs = train_fn(noise, batch)
                outs = dict((k, np.asarray(v)) for k, v in outs.items())
                
                update_dict_of_lists(e_results, **outs)
                u += 1
                pbar.update(u)
            else:
                logger.error('Skipped batch of size {}'.format(batch.shape))
            
        update_dict_of_lists(results, **e_results)
        np.savez(path.join(binary_dir, '{}_results.npz'.format(prefix)),
                 **results)
        
        try:
            if prior == 'uniform':
                noise = floatX(np.random.rand(100, dim_z))
            elif prior == 'gaussian':
                noise = floatX(np.random.normal(size=(100, dim_z)))
            samples = gen_fn(noise)
            summarize(results, samples, image_dir=image_dir,
                      prefix=prefix)
        except Exception as e:
            print(e)
            pass

        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))

        np.savez(path.join(binary_dir, '{}_generator_params.npz'.format(prefix)),
                 *lasagne.layers.get_all_param_values(generator))
        np.savez(path.join(binary_dir,
                           '{}_discriminator_params.npz'.format(prefix)),
                 *lasagne.layers.get_all_param_values(discriminator))
    
    # Load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

_defaults = dict(
    learning_rate=1e-4,
    beta=0.5,
    num_epochs=200,
    batch_size=64,
    dim_z=50,
    prior='gaussian',
    n_samples=20
)

def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-S', '--source', type=str, default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

def setup_out_dir(out_path, name=None):
    if out_path is None:
        raise ValueError('Please set `--out_path` (`-o`) argument.')
    if name is not None:
        out_path = path.join(out_path, name)
        
    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')
    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)
        os.mkdir(binary_dir)
        os.mkdir(image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    kwargs = dict()
    kwargs.update(**_defaults)
    kwargs.update(out_paths)
    logger.info('kwargs: {}'.format(kwargs))
        
    main(source=args.source, **kwargs)
