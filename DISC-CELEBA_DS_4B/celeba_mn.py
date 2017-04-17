#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import datetime
import logging
import os
import sys
import time

import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
import lasagne
from lasagne.layers import (InputLayer, ReshapeLayer,
                                DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid, softmax
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
DIM_X = 64
DIM_Y = 64
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
    def __init__(self, data_stream, img, downsample=True, **kwargs):
        super(To8Bit, self).__init__(
            data_stream=data_stream,
            produces_examples=False,
            **kwargs)
        self.img = img
        self.downsample = downsample
    
    def transform_batch(self, batch):
        if 'features' in self.sources:
            batch = list(batch)
            index = self.sources.index('features')
            new_arr = []
            for arr in batch[index]:
                img = Image.fromarray(arr.transpose(1, 2, 0))
                if self.downsample:
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

def load_stream(batch_size=64, source=None, img=None, downsample=True):
    global DIM_X, DIM_Y
    if downsample:
        DIM_X = 32
        DIM_Y = 32
        
    if source is None:
        raise ValueError('No source provided')
    
    logging.info(
        'Loading data from `{}` {}and quantizing to {} colors').format(
        source, '(downsampling)' if downsampling else '', N_COLORS)
    
    f = h5py.File(source, 'r')
    arr = f['features'][:1000]
    arr = arr.transpose(0, 2, 3, 1)
    arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]))
    img = Image.fromarray(arr).convert(
        'P', palette=Image.ADAPTIVE, colors=N_COLORS)

    train_data = H5PYDataset(path, which_sets=('train',))
    test_data = H5PYDataset(path, which_sets=('test',))
    num_train = train_data.num_examples
    num_test = test_data.num_examples

    train_scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    train_stream = To8Bit(img=img, data_stream=DataStream(
        train_data, iteration_scheme=train_scheme), downsample=downsample)
    test_scheme = ShuffledScheme(examples=num_test, batch_size=batch_size)
    test_stream = To8Bit(img=img, data_stream=DataStream(
        test_data, iteration_scheme=test_scheme), downsample=downsample)
    return train_stream, test_stream, num_train

def print_images(images, num_x, num_y, file='./'):
    scipy.misc.imsave(file,  # current epoch No.
                      (images.reshape(num_x, num_y, DIM_C, DIM_X, DIM_Y)
                       .transpose(0, 3, 1, 4, 2)
                       .reshape(num_x * DIM_X, num_y * DIM_Y, DIM_C)))

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

def reweighted_loss(fake_out):
    log_d1 = -T.nnet.softplus(-fake_out)  # -D_cell.neg_log_prob(1., P=d)
    log_d0 = -(fake_out+T.nnet.softplus(-fake_out))  # -D_cell.neg_log_prob(0., P=d)
    log_w = log_d1 - log_d0

    # Find normalized weights.
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N

    cost = ((log_w - T.maximum(log_Z_est, -2)) ** 2).mean()

    return cost

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

def train(num_epochs,
          filename,
          gen_lr=1e-4,
          beta_1_gen=0.5,
          beta_1_disc=0.5,
          print_freq=50,
          disc_lr=1e-4,
          num_iter_gen=1,
          n_samples=20,
          image_dir=None,
          binary_dir=None,
          gt_image_dir=None,
          source=None):
    
    # Load the dataset
    log_file = open(filename, 'w')
    print("Loading data...")
    print("Testing RW_DCGAN ...")
    log_file.write("Testing RW_DCGAN...\n")
    log_file.write("Loading data...\n")
    log_file.write("Num_epochs: {}, disc_lr: {}, gen_lr: {}\n".format(
        num_epochs, disc_lr, gen_lr))
    log_file.flush()
    train_stream, test_stream = load_stream(img=img)
    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    #target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling GAN functions...")
    log_file.write("Building model and compiling GAN functions...\n")
    parameter = initial_parameters()
    generator = build_generator(parameter, noise_var)
    discriminator = build_discriminator(input_var)
    trng = RandomStreams(random.randint(1, 1000000))

    # Sample
    batch_size = noise_var.shape[0]
    dim_x = input_var.shape[2]
    dim_y = input_var.shape[3]

    g_output_logit = lasagne.layers.get_output(generator)
    g_output_logit_ = g_output_logit.transpose(0, 2, 3, 1)
    g_output_logit_ = g_output_logit_.reshape((-1, 16))
    g_output = T.nnet.softmax(g_output_logit_)
    g_output = g_output.reshape((-1, dim_x, dim_y, 16))
    p_p = T.shape_padleft(g_output)
    g_output = g_output.transpose(0, 3, 1, 2)
    p_t = T.zeros((n_samples, batch_size, dim_x, dim_y, 16)) + p_p
    p = p_t.reshape((-1, 16))
    samples = trng.multinomial(pvals=p).astype(floatX)
    samples = theano.gradient.disconnected_grad(samples)
    samples = samples.reshape((n_samples, batch_size, dim_x, dim_y, 16))
    samples = samples.transpose(0, 1, 4, 2, 3)
    #samples = (R <= T.shape_padleft(g_output)).astype(floatX)
    
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, samples.reshape(
            (n_samples * batch_size, 16, dim_x, dim_y)))
    fake_out_ = fake_out.reshape((n_samples, batch_size))
    
    log_d1 = -T.nnet.softplus(-fake_out_)
    log_d0 = -(fake_out_ + T.nnet.softplus(-fake_out_))
    #log_w = log_d1 - log_d0
    log_w = fake_out_
    #g_output_ = T.shape_padleft(T.clip(g_output, 1e-7, 1. - 1e-7))
    #log_g = (samples * T.log(g_output_) + (1. - samples) * T.log(1. - g_output_)).sum(axis=(2, 3, 4))
    #log_g = (samples * T.log(g_output) + (1. - samples) * T.log(1. - g_output)).sum(axis=(2, 3, 4))
    #log_g = (samples * -T.nnet.softplus(-g_output_logit) + (1. - samples) * -(g_output_logit + T.nnet.softplus(-g_output_logit))).sum(axis=(2, 3, 4))
    #log_g = (-T.nnet.softplus(-g_output_logit) - (1. - samples) * g_output_logit).sum(axis=(2, 3, 4))
    log_g = (samples * (g_output_logit - log_sum_exp2(g_output_logit, axis=1))[None, :, :, :, :]).sum(axis=(2, 3, 4))
    
    # Find normalized weights.
    log_N = T.log(log_w.shape[0]).astype(floatX)
    #log_Z_est = T.maximum(log_sum_exp(log_w - log_N, axis=0), -4)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est_ = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)

    #Create gen_loss
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    #generator_loss = (T.nnet.softplus(-fake_out)).mean() -- Original GAN loss

    # Create disc_loss
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (T.nnet.softplus(-fake_out)).mean() + fake_out.mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    # Losses / updates    
    generator_updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=gen_lr, beta1=beta_1_gen)
    discriminator_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=disc_lr,
        beta1=beta_1_disc)

    '''
    generator_updates = lasagne.updates.rmsprop(
        generator_loss, generator_params, learning_rate=gen_lr)
    discriminator_updates = lasagne.updates.rmsprop(
        discriminator_loss, discriminator_params, learning_rate=disc_lr)
    '''
    disc_stats = {
        'D loss': discriminator_loss,
        'P(real)': (real_out > 0.).mean()
    }
    
    gen_stats = {
        'G loss': generator_loss,
        'P(fake)': (fake_out < 0.).mean(),
        'Z (est)': log_Z_est_.mean(),
        'log D(fake=1)': log_d1.mean(),
        'log D(fake=0)': log_d0.mean(),
        'log g': log_g.mean(),
        'norm w': w_tilde.mean(),
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean()
    }

    train_discriminator = theano.function([noise_var, input_var],
                               disc_stats, allow_input_downcast=True,
                               updates=discriminator_updates)

    train_generator = theano.function([noise_var, input_var],
                               gen_stats, allow_input_downcast=True,
                               updates=generator_updates)

    # Compile another function generating some data

    g_output_s = lasagne.layers.get_output(generator, deterministic=True)
    gen_fn = theano.function([noise_var], g_output_s) 

    f_test = theano.function([noise_var, input_var], [g_output[0, :, 10, 10], samples[:, 0, :, 10, 10]])
    
    # Finally, launch the training loop.
    print("Starting training of GAN...")
    log_file.write("Starting training of GAN...\n")
    log_file.flush()
    # We iterate over epochs:

    samples = gen_fn(lasagne.utils.floatX(np.random.rand(50, 100)))
    print(samples.size)
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print("Epoch: ", epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        prefix = "ep_{}".format(epoch)
        
        '''
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=(50000 // 64)).start()
        '''
        
        for batch in train_stream.get_epoch_iterator():
            inputs = np.array(batch[0], dtype=np.float32)
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))

            #print(f_test(noise, inputs))
            
            #samples_print_gt = convert_to_rgb(inputs, img)
            #print_images(samples_print_gt[:64], 8, 8, file=gt_image_dir + prefix + '_gt.png')
            
            train_discriminator(noise, inputs)
            disc_outs = train_discriminator(noise, inputs) 
            gen_outs = {}

            for i in range(num_iter_gen):
                gen_train_out = train_generator(noise, inputs)
                gen_outs_ = gen_train_out
                if gen_outs == {}:
                    gen_outs = gen_outs_
                else:
                    for k in gen_outs.keys():
                        gen_outs[k].append(gen_outs_[k])
                
            for k, v in gen_outs.items():
                gen_outs[k] = np.mean(v)

            train_batches += 1
            if train_batches % print_freq == 0:
                print('-' * 80)
                print("Batch Number: {}, Epoch Number: {}".format(train_batches + 1, epoch + 1))
                found_nan = False
                for k, v in disc_outs.items():
                    print('{0}: {1}'.format(k, v))
                    if np.isnan(v):
                        found_nan = True
                for k, v in gen_outs.items():
                    print('{0}: {1}'.format(k, v))
                    if np.isnan(v):
                        found_nan = True
                if found_nan:
                    print('Found NAN')
                    break
                '''
                log_file.write('-' * 80 + '\n')
                log_file.write("Batch Number: {}".format(train_batches + 1, epoch + 1) + '\n')
                log_file.write("Generator: p_fake: {}, gen_loss: {} \n".format(p_fake, disc_loss))
                log_file.write("Discriminator: p_real: {}, disc_loss: {} \n".format(p_real, disc_loss))
                log_file.write('-' * 80 + '\n')
                '''
                samples = gen_fn(lasagne.utils.floatX(np.random.rand(49, 100)))
                samples_print = convert_to_rgb(samples, img)
                print_images(samples_print, 7, 7, file=image_dir + prefix + "_{}".format(train_batches) +'_gen.png')
                
                #samples_print_gt = convert_to_rgb(inputs, img)
                #print_images(samples_print_gt[:64], 8, 8, file=gt_image_dir + prefix + '_gt.png')

        # Then we print the results for this epoch:
        print("Total Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        log_file.write("Total Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))
        log_file.write("  training loss:\t{}\n".format(train_err / train_batches))
        log_file.flush()

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(49, 100)))
        samples_print = convert_to_rgb(samples, img)
        print_images(samples_print, 7, 7, file=image_dir + prefix + '_gen.png')
        #if epoch == num_epochs - 1: #save binary data for further calculation
        np.savez(binary_dir + prefix + '_celeba_gen_params.npz',
                 *lasagne.layers.get_all_param_values(generator))

    log_file.flush()
    log_file.close()



