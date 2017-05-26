#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import sys
import os
from os import path
import time

import lasagne
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
from lasagne.nonlinearities import sigmoid, LeakyRectify, sigmoid
from lasagne.layers import (
    batch_norm, DenseLayer, InputLayer, ReshapeLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
from matplotlib import pyplot as plt
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano
import theano.tensor as T


DIM_X = 28
DIM_Y = 28
DIM_C = 1
lrelu = LeakyRectify(0.2)
floatX = lasagne.utils.floatX

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

# ################## DATA ##################

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
##################### MODEL #######################

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                                (self.input_shape[1], num_filters) + self.filter_size,
                                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
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

def build_generator(input_var=None, dim_z=100):
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 128 * 7 * 7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2,
                          nonlinearity=sigmoid)
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer


def build_discriminator(input_var=None):
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer

# ############################# MATH ###############################

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
    log_factor = log_factor - T.log(log_factor.shape[0]).astype('float32')
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

# ############################# LOSSES ###############################

def BGAN(fake_out, real_out, log_Z, use_Z=True):    
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
        
    if use_Z:
        generator_loss = ((fake_out - log_Z) ** 2).mean()
    else:
        generator_loss = (fake_out ** 2).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        T.nnet.softplus(-fake_out) + fake_out).mean()
    return generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est

def GAN(fake_out, real_out):
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
         
    generator_loss = T.nnet.softplus(-fake_out).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        T.nnet.softplus(-fake_out) + fake_out).mean()
    return generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est

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

def main(num_epochs=None, method=None, batch_size=None,
         learning_rate=None, beta=None,
         image_dir=None, binary_dir=None,
         prior=None, dim_z=None):
    
    # DATA
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    train_samples = X_train.shape[0]

    # VAR
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')

    # MODEL
    logger.info('Building model and graph')
    generator = build_generator(noise_var, dim_z=dim_z)
    discriminator = build_discriminator(input_var)

    # GRAPH
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, lasagne.layers.get_output(generator))

    if method == 'BGAN':
        generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est = BGAN(
            fake_out, real_out, log_Z)
        other_loss, _, _, _, _ = GAN(fake_out, real_out)
    elif method == 'GAN':
        generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est = GAN(
            fake_out, real_out)
        other_loss, _, _, _, _ = BGAN(fake_out, real_out, log_Z)
    else:
        raise NotImplementedError('Unsupported method `{}`'.format(method))

    # OPTIMIZER
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator,
                                                         trainable=True)
    
    eta = theano.shared(lasagne.utils.floatX(learning_rate))
    
    updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=eta, beta1=beta)
    updates.update(lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=eta, beta1=beta))
    updates.update([(log_Z, 0.95 * log_Z + 0.05 * log_Z_est.mean())])

    # COMPILE
    results = {
        'p(real)': (T.nnet.sigmoid(real_out) > .5).mean(),
        'p(fake': (T.nnet.sigmoid(fake_out) < .5).mean(),
        'G loss': generator_loss,
        'Other loss': other_loss,
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
        [noise_var], lasagne.layers.get_output(generator, deterministic=True))

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
        for batch in iterate_minibatches(X_train, y_train, batch_size,
                                         shuffle=True):
            inputs, targets = batch
            if batch0 is None: batch0 = inputs
            
            if prior == 'uniform':
                noise = floatX(np.random.rand(len(inputs), dim_z))
            elif prior == 'gaussian':
                noise = floatX(numpy.random.normal(size=(len(inputs), dim_z)))
                
            outs = train_fn(noise, inputs)
            outs = dict((k, np.asarray(v)) for k, v in outs.items())
            
            update_dict_of_lists(e_results, **outs)
            u += 1
            pbar.update(u)
            
        update_dict_of_lists(results, **e_results)
        np.savez(path.join(binary_dir, '{}_results.npz'.format(prefix)),
                 **results)
            
        try:
            if prior == 'uniform':
                noise = floatX(np.random.rand(100, dim_z))
            elif prior == 'gaussian':
                noise = floatX(numpy.random.normal(size=(64, dim_z)))
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
    learning_rate=1e-2,
    beta=0.5,
    num_epochs=200,
    batch_size=64,
    method='BGAN',
    dim_z=10,
    prior='gaussian'
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
        
    main(**kwargs)
