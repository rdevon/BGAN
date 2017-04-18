#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import argparse
import datetime
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
import numpy as np
from PIL import Image
from progressbar import Bar, ProgressBar, Percentage, Timer
import pylab as pl
import random
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc


floatX = theano.config.floatX
data_name = "celeba_64.hdf5"
lrelu = LeakyRectify(0.2)

N_COLORS = 16
DIM_X = 32
DIM_Y = 32
DIM_C = 3

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

class To8Bit(Transformer):
    def __init__(self, data_stream, img, **kwargs):
        super(To8Bit, self).__init__(
            data_stream=data_stream,
            produces_examples=False,
            **kwargs)
        self.img = img
    
    def transform_batch(self, batch):
        if 'features' in self.sources:
            batch = list(batch)
            index = self.sources.index('features')
            new_arr = []
            for arr in batch[index]:
                img = Image.fromarray(arr.transpose(1, 2, 0))
                if img.size[0] != DIM_X:
                    img = img.resize((DIM_X, DIM_Y), Image.ANTIALIAS)
                img = img.quantize(palette=self.img, colors=N_COLORS)
                arr = np.array(img)
                arr[arr > (N_COLORS - 1)] = 0 #HACK don't know why it's giving more colors sometimes
                
                arr_ = np.zeros((N_COLORS, DIM_X * DIM_Y)).astype(arr.dtype)
                arr_[arr.flatten(), np.arange(DIM_X * DIM_Y)] = 1
                new_arr.append(arr_.reshape((N_COLORS, DIM_X, DIM_Y)))
            batch[index] = np.array(new_arr)
            batch = tuple(batch)
        return batch

def load_stream(batch_size=64, source=None, img=None):
    if source is None:
        raise ValueError('No source provided')
    
    logger.info(
        'Loading data from `{}` (using {}x{}) and quantizing to {} colors'.format(
        source, DIM_X, DIM_Y, N_COLORS))
    
    f = h5py.File(source, 'r')
    arr = f['features'][:1000]
    arr = arr.transpose(0, 2, 3, 1)
    arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]))
    img = Image.fromarray(arr).convert(
        'P', palette=Image.ADAPTIVE, colors=N_COLORS)

    train_data = H5PYDataset(source, which_sets=('train',))
    num_train = train_data.num_examples

    train_scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    train_stream = To8Bit(img=img, data_stream=DataStream(
        train_data, iteration_scheme=train_scheme))
    return train_stream, num_train, img

def print_images(images, num_x, num_y, file='./'):
    scipy.misc.imsave(file,  # current epoch No.
                      (images.reshape(num_x, num_y, DIM_C, DIM_X, DIM_Y)
                       .transpose(0, 3, 1, 4, 2)
                       .reshape(num_x * DIM_X, num_y * DIM_Y, DIM_C)))
    
def convert_to_rgb(samples, img):
    new_samples = []
    samples = samples.argmax(axis=1).astype('uint8')
    for sample in samples:
        img2 = Image.fromarray(sample)
        img2.putpalette(img.getpalette())
        img2 = img2.convert('RGB')
        new_samples.append(np.array(img2))
    samples_print = np.array(new_samples).transpose(0, 3, 1, 2)
    return samples_print

# ##################### MODEL #######################

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 W=None, b=None,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        if W is None:
            self.W = self.add_param(
                lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        else:
            self.W = self.add_param(
                W,
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

def build_discriminator(input_var=None, dim_h=128):
    layer = InputLayer(shape=(None, N_COLORS, DIM_X, DIM_Y), input_var=input_var)
    layer = Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv2DLayer(layer, dim_h * 4, 5, stride=2, pad=2, nonlinearity=lrelu)
    if DIM_X == 64:
        layer = Conv2DLayer(layer, dim_h * 4, 5, stride=2, pad=2,
                            nonlinearity=lrelu)
    
    layer = DenseLayer(layer, 1, nonlinearity=None)

    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer

def build_generator(input_var=None, dim_h=128):
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, dim_h * 4 * 4 * 4))
    layer = ReshapeLayer(layer, ([0], dim_h * 4, 4, 4))
    layer = batch_norm(Deconv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    if DIM_X == 64:
        layer = batch_norm(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, N_COLORS, 5, stride=2, pad=2,
                          nonlinearity=None)

    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer


# ##################### MATH #######################

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def log_sum_exp2(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis, keepdims=True)
    return y

def norm_exp(log_factor):
    '''Gets normalized weights.

    '''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

# ##################### LOSS #####################

def BGAN(discriminator, g_output_logit, n_samples, trng, batch_size=64):
    d = OrderedDict()
    
    d['g_output_logit'] = g_output_logit
    g_output_logit_ = g_output_logit.transpose(0, 2, 3, 1)
    g_output_logit_ = g_output_logit_.reshape((-1, N_COLORS))
    d['g_output_logit_'] = g_output_logit_
    
    g_output = T.nnet.softmax(g_output_logit_)
    g_output = g_output.reshape((batch_size, DIM_X, DIM_Y, N_COLORS))
    d['g_output'] = g_output
    
    p_t = T.tile(T.shape_padleft(g_output), (n_samples, 1, 1, 1, 1))
    d['p_t'] = p_t
    p = p_t.reshape((-1, N_COLORS))
    d['p'] = p
    
    samples = trng.multinomial(pvals=p).astype(floatX)
    samples = theano.gradient.disconnected_grad(samples)
    samples = samples.reshape((n_samples, batch_size, DIM_X, DIM_Y, N_COLORS))
    samples = samples.transpose(0, 1, 4, 2, 3)
    d['samples'] = samples
    
    D_r = lasagne.layers.get_output(discriminator)
    D_f = lasagne.layers.get_output(
        discriminator,
        samples.reshape((-1, N_COLORS, DIM_X, DIM_Y)))
    D_f_ = D_f.reshape((n_samples, -1))
    d.update(D_r=D_r, D_f=D_f, D_f_=D_f_)
    
    log_d1 = -T.nnet.softplus(-D_f_)
    log_d0 = -(D_f_ + T.nnet.softplus(-D_f_))
    log_w = D_f_
    d.update(log_d1=log_d1, log_d0=log_d0, log_w=log_w)

    log_g = (samples * (g_output_logit - log_sum_exp2(
        g_output_logit, axis=1))[None, :, :, :, :]).sum(axis=(2, 3, 4))
    d['log_g'] = log_g
    
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
    d.update(log_w_tilde=log_w_tilde, w_tilde=w_tilde)
    
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-D_r)).mean() + (
        T.nnet.softplus(-D_f)).mean() + D_f.mean()

    return generator_loss, discriminator_loss, D_r, D_f, log_Z_est, w_tilde, d


# ##################### MAIN #####################

def summarize(results, samples, gt_samples, train_batches=None,
              image_dir=None, gt_image_dir=None, prefix='', img=None):
    results = dict((k, np.mean(v)) for k, v in results.items())    
    logger.info(results)
    if image_dir is not None:
        samples_print = convert_to_rgb(samples, img)
        print_images(samples_print, 8, 8, file=path.join(
            image_dir, '{}_{}_gen.png'.format(prefix, train_batches)))
    
    if gt_image_dir is not None:
        samples_print_gt = convert_to_rgb(gt_samples, img)
        print_images(samples_print_gt, 8, 8, file=path.join(
            gt_image_dir, '{}_{}_gt.png'.format(prefix, train_batches)))

def main(source=None, num_epochs=None,
         learning_rate=None, beta=None,
         dim_noise=None, batch_size=None, n_samples=None,
         image_dir=None, binary_dir=None, gt_image_dir=None,
         summary_updates=100, debug=False):
    
    # Load the dataset
    stream, train_samples, img = load_stream(
        source=source, batch_size=batch_size)
    
    # VAR
    noise = T.matrix('noise')
    input_var = T.tensor4('inputs')
    
    # MODELS
    generator = build_generator(noise)
    discriminator = build_discriminator(input_var)
    
    trng = RandomStreams(random.randint(1, 1000000))

    # GRAPH / LOSS    
    g_output_logit = lasagne.layers.get_output(generator)
    generator_loss, discriminator_loss, D_r, D_f, log_Z_est, w_tilde, d = BGAN(
        discriminator, g_output_logit, n_samples, trng)

    if debug:
        batch = stream.get_epoch_iterator().next()[0]
        noise_ = lasagne.utils.floatX(np.random.rand(batch.shape[0],
                                                     dim_noise))
        print(batch.shape)
        for k, v in d.items():
            print('Testing {}'.format(k))
            f = theano.function([noise, input_var], v, on_unused_input='warn')
            print(k, f(noise_, batch.astype(floatX)).shape)

    # OPTIMIZER
    discriminator_params = lasagne.layers.get_all_params(
        discriminator, trainable=True)
    generator_params = lasagne.layers.get_all_params(
        generator, trainable=True)
    
    l_kwargs = dict(learning_rate=learning_rate, beta1=beta)
    
    d_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, **l_kwargs)
    g_updates = lasagne.updates.adam(
        generator_loss, generator_params, **l_kwargs)
    
    outputs = {
        'G cost': generator_loss,
        'D cost': discriminator_loss,
        'p(real)': T.nnet.sigmoid(D_r > .5).mean(),
        'p(fake)': T.nnet.sigmoid(D_f > .5).mean(),
        'Z (est)': log_Z_est.mean(),
        'norm w': w_tilde.mean(),
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean()
    }
    gen_train_fn = theano.function([input_var, noise], outputs, updates=g_updates)
    disc_train_fn = theano.function([input_var, noise], [], updates=d_updates)
    gen_fn = theano.function([noise], lasagne.layers.get_output(
        generator, deterministic=True))

    # train
    logger.info('Starting training...')
    
    for epoch in range(num_epochs):
        train_batches = 0
        start_time = time.time()
        
        # Train
        u = 0
        results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        prefix = str(epoch)
        
        for batch in stream.get_epoch_iterator():
            noise = lasagne.utils.floatX(np.random.rand(batch[0].shape[0],
                                                        dim_noise))
            disc_train_fn(batch[0].astype(floatX), noise)
            outs = gen_train_fn(batch[0].astype(floatX), noise)
            update_dict_of_lists(results, **outs)
            
            if u % summary_updates == 0:
                try:
                    samples = gen_fn(lasagne.utils.floatX(
                        np.random.rand(64, dim_noise)))
                    summarize(results, samples, batch[0][:64], train_batches=u,
                              image_dir=image_dir, gt_image_dir=gt_image_dir,
                              img=img, prefix=prefix)
                except Exception as e:
                    print(e)
                    pass
            
            u += 1
            pbar.update(u)
                
        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))


        # And finally, we plot some generated data
        ssamples = gen_fn(lasagne.utils.floatX(
            np.random.rand(64, dim_noise)))
        summarize(results, samples, None,
                  image_dir=image_dir, train_batches='final',
                  img=img, prefix=prefix)
        np.savez(path.join(binary_dir, '{}_celeba_gen_params.npz'.format(prefix)),
                 *lasagne.layers.get_all_param_values(generator))

_defaults = dict(
    learning_rate=1e-3,
    beta=0.5,
    num_epochs=100,
    dim_noise=100,
    batch_size=64,
    n_samples=20,
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
    parser.add_argument('-S', '--source', type=str, default=None)
    parser.add_argument('-V', '--vocab', type=str, default=None)
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
    gt_image_dir = path.join(out_path, 'gt_images')
    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)
        os.mkdir(binary_dir)
        os.mkdir(image_dir)
        os.mkdir(gt_image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir, gt_image_dir=gt_image_dir)

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