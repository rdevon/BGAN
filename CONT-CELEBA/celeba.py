#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import datetime
import logging
import os
from os import path
import sys
import time

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import h5py
import lasagne
from lasagne.layers import (InputLayer, ReshapeLayer,
                                DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import pylab as pl
import theano
import theano.tensor as T
import scipy.misc


floatX = theano.config.floatX

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

# ##################### DATA #####################

def load_stream(batch_size=None, source=None):
    logger.info('Loading data from `{}`'.format(source))
    
    train_data = H5PYDataset(source, which_sets=('train',))
    test_data = H5PYDataset(source, which_sets=('test',))

    num_train = train_data.num_examples
    num_test = test_data.num_examples

    logger.debug('Number of test examples: {}'.format(num_test))
    logger.debug('Number of training examples: {}'.format(num_train))
    
    train_scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    train_stream = DataStream(train_data, iteration_scheme=train_scheme)
    test_scheme = ShuffledScheme(examples=num_test, batch_size=batch_size)
    test_stream = DataStream(test_data, iteration_scheme=test_scheme)
    return train_stream, test_stream

def transform(image):
    return np.array(image) / 127.5 - 1.  # seems like normalization

def inverse_transform(image):
    return (np.array(image) + 1.) * 127.5

# ##################### Build the neural network model #######################

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

def build_discriminator(input_var=None):

    lrelu = LeakyRectify(0.2)
    layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    layer = Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = batch_norm(Conv2DLayer(layer, 128 * 2, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 4, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 8, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    logger.debug('Discriminator output: {}' .format(layer.output_shape))
    return layer


def initial_parameters():
    parameter = {}
    parameter['beta1'] = theano.shared(np.zeros(128 * 8 * 4 * 4).astype('float32'))
    parameter['gamma1'] = theano.shared(np.ones(128 * 8 * 4 * 4).astype('float32'))
    parameter['mean1'] = theano.shared(np.zeros(128 * 8 * 4 * 4).astype('float32'))
    parameter['inv_std1'] = theano.shared(np.ones(128 * 8 * 4 * 4).astype('float32'))
    parameter['beta2'] = theano.shared(np.zeros(128 * 4).astype('float32'))
    parameter['gamma2'] = theano.shared(np.ones(128 * 4).astype('float32'))
    parameter['mean2'] = theano.shared(np.zeros(128 * 4).astype('float32'))
    parameter['inv_std2'] = theano.shared(np.ones(128 * 4).astype('float32'))
    parameter['beta3'] = theano.shared(np.zeros(128 * 2).astype('float32'))
    parameter['gamma3'] = theano.shared(np.ones(128 * 2).astype('float32'))
    parameter['mean3'] = theano.shared(np.zeros(128 * 2).astype('float32'))
    parameter['inv_std3'] = theano.shared(np.ones(128 * 2).astype('float32'))
    parameter['beta4'] = theano.shared(np.zeros(128).astype('float32'))
    parameter['gamma4'] = theano.shared(np.ones(128).astype('float32'))
    parameter['mean4'] = theano.shared(np.zeros(128).astype('float32'))
    parameter['inv_std4'] = theano.shared(np.ones(128).astype('float32'))
    return parameter


def build_generator(parameter, input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import tanh

    layer = InputLayer(shape=(None, 100), input_var=input_var)
    layer = DenseLayer(layer, 128 * 8 * 4 * 4)
    parameter['W1'] = layer.W
    parameter['b1'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta1'], gamma=parameter['gamma1'],
                       mean=parameter['mean1'], inv_std=parameter['inv_std1'])
    layer = ReshapeLayer(layer, ([0], 128 * 8, 4, 4))
    layer = Deconv2DLayer(layer, 128 * 4, 5, stride=2, pad=2)
    parameter['W2'] = layer.W
    parameter['b2'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta2'], gamma=parameter['gamma2'],
                       mean=parameter['mean2'], inv_std=parameter['inv_std2'])

    layer = Deconv2DLayer(layer, 128 * 2, 5, stride=2, pad=2)
    parameter['W3'] = layer.W
    parameter['b3'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta3'], gamma=parameter['gamma3'],
                       mean=parameter['mean3'], inv_std=parameter['inv_std3'])
    layer = Deconv2DLayer(layer, 128, 5, stride=2, pad=2)
    parameter['W4'] = layer.W
    parameter['b4'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta4'], gamma=parameter['gamma4'],
                       mean=parameter['mean4'], inv_std=parameter['inv_std4'])

    layer = Deconv2DLayer(layer, 3, 5, stride=2, pad=2, nonlinearity=tanh)
    parameter['W5'] = layer.W
    parameter['b5'] = layer.b
    
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def print_images(images, num_x, num_y, file='./'):
    scipy.misc.imsave(file,  # current epoch No.
                      (images.reshape(num_x, num_y, 3, 64, 64)
                       .transpose(0, 3, 1, 4, 2)
                       .reshape(num_x * 64, num_y * 64, 3)))

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
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

def BGAN(fake_out, real_out, log_Z):    
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
         
    generator_loss = ((fake_out - log_Z) ** 2).mean()
    #generator_loss = (fake_out ** 2).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    return generator_loss, discriminator_loss, log_Z_est

def GAN(fake_out, real_out):
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    # Find normalized weights.
    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
         
    generator_loss = T.nnet.softplus(-fake_out).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    return generator_loss, discriminator_loss, log_Z_est

def train(num_epochs,
          filename,
          batch_size=64,
          gen_lr=1e-3,
          beta_1_gen=0.5,
          beta_1_disc=0.5,
          print_freq=200,
          disc_lr=1e-3,
          num_iter_gen=1,
          image_dir=None,
          binary_dir=None):
    
    set_stream_logger(2)
    set_file_logger(path.join(binary_dir, 'out.log'))

    logger.info('Num_epochs: {}, disc_lr: {}, gen_lr: {}\n'.format(
        num_epochs, disc_lr, gen_lr))
    
    # Load the dataset
    source = '/home/devon/Data/basic/celeba_64.hdf5'
    f = h5py.File(source, 'r')
    arr = f['features']
    training_samples = arr.shape[0]
    
    train_stream, test_stream = load_stream(source=source,
                                            batch_size=batch_size)
    
    # Input vars
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')

    # Create neural network model
    logger.info('Building model and compiling GAN functions...')
    parameter = initial_parameters()
    generator = build_generator(parameter, noise_var)
    discriminator = build_discriminator(input_var)

    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))

    generator_loss, discriminator_loss, log_Z_est = BGAN(fake_out, real_out, log_Z) 

    # Optimizer
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    generator_updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=gen_lr, beta1=beta_1_gen)
    generator_updates.update([(log_Z, 0.95 * log_Z + 0.05 * log_Z_est.mean())])
    discriminator_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=disc_lr, beta1=beta_1_disc)

    d_results = {
        'p(real)': (real_out > 0.).mean(),
        'L_D': discriminator_loss
    }
    
    g_results = {
        'p(fake)': (fake_out < 0.).mean(),
        'L_G': generator_loss,
        'log Z': log_Z,
        'log Z (est)': log_Z_est.mean()
    }

    train_discriminator = theano.function([noise_var, input_var],
        d_results, allow_input_downcast=True, updates=discriminator_updates)

    train_generator = theano.function([noise_var],
        g_results, allow_input_downcast=True, updates=generator_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    logger.info('Starting training of GAN...')
    # We iterate over epochs:
    for epoch in range(num_epochs):
        logger.info('Epoch: '.format(epoch))
        train_batches = 0
        start_time = time.time()
        prefix = 'ep_{}'.format(epoch)
        
        results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(training_samples // batch_size)).start()
        
        for batch in train_stream.get_epoch_iterator():
            inputs = transform(np.array(batch[0],dtype=np.float32))  # or batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))

            train_discriminator(noise, inputs)
            disc_train_out = train_discriminator(noise, inputs)
            d_outs = disc_train_out
            update_dict_of_lists(results, **d_outs)

            for i in range(num_iter_gen):
                g_outs = train_generator(noise)
                g_outs = dict((k, np.asarray(v)) for k, v in g_outs.items())
                update_dict_of_lists(results, **g_outs)

            train_batches += 1
            pbar.update(train_batches)
            
            if train_batches % print_freq == 0:
                result_summary = dict((k, np.mean(v)) for k, v in results.items())
                logger.info(result_summary)
                
                samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
                samples_print = samples[0:49]
                print_images(inverse_transform(samples_print), 7, 7, file=image_dir + prefix + '_gen_tmp.png')

        # Then we print the results for this epoch:
        logger.info('Total Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
        samples_print = samples[0:49]
        print_images(inverse_transform(samples_print), 7, 7, file=image_dir + prefix + '_gen.png')
        np.savez(binary_dir + prefix + '_celeba_gen_params.npz', *lasagne.layers.get_all_param_values(generator))


    log_file.flush()
    log_file.close()



