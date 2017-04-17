#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import datetime
import sys
import time
import os

import h5py
from progressbar import Bar, ProgressBar, Percentage, Timer
import numpy as np
import random
import theano
import theano.tensor as T
from PIL import Image
import pylab as pl
import lasagne
import scipy.misc
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

from lasagne.layers import (InputLayer, ReshapeLayer,
                                DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
from PIL import Image

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


floatX = theano.config.floatX
data_name = "celeba_64.hdf5"


class To8Bit(Transformer):
    def __init__(self, data_stream, img, **kwargs):
        self.img = img
        super(To8Bit, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)
        
    def transform_example(self, example):
        if 'features' in self.sources:
            example = list(example)
            index = self.sources.index('features')
            arr = example[index]
            img = Image.fromarray(arr.transpose(1, 2, 0))
            #img = img.resize((32, 32), Image.ANTIALIAS)
            img = img.convert(mode='P', palette=Image.WEB, colors=16)
            arr = np.array(img)
            example[index] = (((arr[:, :, None] & (1 << np.arange(4)))) > 0).astype(int).transpose(2, 0, 1)
            example = tuple(example)
        return example
    
    def transform_batch(self, batch):
        if 'features' in self.sources:
            batch = list(batch)
            index = self.sources.index('features')
            new_arr = []
            for arr in batch[index]:
                #print(arr[:, 0, 0])
                img = Image.fromarray(arr.transpose(1, 2, 0))
                img = img.resize((32, 32), Image.ANTIALIAS)
                img = img.quantize(palette=self.img)
                arr = np.array(img)
                new_arr.append((((arr[:, :, None] & (1 << np.arange(4)))) > 0).astype(int))
            batch[index] = np.array(new_arr).transpose(0, 3, 1, 2)
            batch = tuple(batch)
        return batch

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

#def load_stream(batch_size=128, path="/data/lisa/data/", img=None):
def load_stream(batch_size=64, path="/home/devon/Data/basic/", img=None):
    path = os.path.join(path, data_name)
    train_data = H5PYDataset(path, which_sets=('train',))
    test_data = H5PYDataset(path, which_sets=('test',))
    num_train = train_data.num_examples
    num_test = test_data.num_examples
    print("Number of test examples: ", num_test)
    print("Number of training examples: ", num_train)
    train_scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    train_stream = To8Bit(img=img, data_stream=DataStream(
        train_data, iteration_scheme=train_scheme))
    test_scheme = ShuffledScheme(examples=num_test, batch_size=batch_size)
    test_stream = To8Bit(img=img, data_stream=DataStream(
        test_data, iteration_scheme=test_scheme))
    return train_stream, test_stream

def transform(image):
    return np.array(image) / 127.5 - 1.  # seems like normalization

def inverse_transform(image):
    return (np.array(image) + 1.) * 127.5

# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

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
    layer = InputLayer(shape=(None, 4, 32, 32), input_var=input_var)
    # two convolutions
    layer = Conv2DLayer(layer, 256, 5, stride=2, pad=2, nonlinearity=lrelu)
    
    '''
    layer = batch_norm(Conv2DLayer(layer, 128 * 2, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 4, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 8, 5, stride=2, pad=2, nonlinearity=lrelu))
    '''
    
    layer = Conv2DLayer(layer, 256 * 2, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv2DLayer(layer, 256 * 4, 5, stride=2, pad=2, nonlinearity=lrelu)
    #layer = Conv2DLayer(layer, 128 * 8, 5, stride=2, pad=2, nonlinearity=lrelu)
    
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print("Discriminator output:", layer.output_shape)
    return layer


def initial_parameters():
    parameter = {}
    parameter['beta1'] = theano.shared(np.zeros(256 * 4 * 4 * 4).astype('float32'))
    parameter['gamma1'] = theano.shared(np.ones(256 * 4 * 4 * 4).astype('float32'))
    parameter['mean1'] = theano.shared(np.zeros(256 * 4 * 4 * 4).astype('float32'))
    parameter['inv_std1'] = theano.shared(np.ones(256 * 4 * 4 * 4).astype('float32'))
    parameter['beta2'] = theano.shared(np.zeros(256 * 2).astype('float32'))
    parameter['gamma2'] = theano.shared(np.ones(256 * 2).astype('float32'))
    parameter['mean2'] = theano.shared(np.zeros(256 * 2).astype('float32'))
    parameter['inv_std2'] = theano.shared(np.ones(256 * 2).astype('float32'))
    parameter['beta3'] = theano.shared(np.zeros(256).astype('float32'))
    parameter['gamma3'] = theano.shared(np.ones(256).astype('float32'))
    parameter['mean3'] = theano.shared(np.zeros(256).astype('float32'))
    parameter['inv_std3'] = theano.shared(np.ones(256).astype('float32'))
    return parameter


def build_generator(parameter, input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import tanh, sigmoid

    layer = InputLayer(shape=(None, 100), input_var=input_var)

    # fully-connected layer
    layer = DenseLayer(layer, 256 * 4 * 4 * 4)
    parameter['W1'] = layer.W
    parameter['b1'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta1'], gamma=parameter['gamma1'],
                       mean=parameter['mean1'], inv_std=parameter['inv_std1'])
    layer = ReshapeLayer(layer, ([0], 256 * 4, 4, 4))
    layer = Deconv2DLayer(layer, 256 * 2, 5, stride=2, pad=2)
    parameter['W2'] = layer.W
    parameter['b2'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta2'], gamma=parameter['gamma2'],
                       mean=parameter['mean2'], inv_std=parameter['inv_std2'])

    layer = Deconv2DLayer(layer, 256, 5, stride=2, pad=2)
    parameter['W3'] = layer.W
    parameter['b3'] = layer.b
    layer = batch_norm(layer, beta=parameter['beta3'], gamma=parameter['gamma3'],
                       mean=parameter['mean3'], inv_std=parameter['inv_std3'])
    
    layer = Deconv2DLayer(layer, 4, 5, stride=2, pad=2, nonlinearity=sigmoid)
    parameter['W4'] = layer.W
    parameter['b4'] = layer.b
    # shape=(batch,1,28,28)
    print("Generator output:", layer.output_shape)
    return layer

def print_images(images, num_x, num_y, file='./'):
    scipy.misc.imsave(file,  # current epoch No.
                      (images.reshape(num_x, num_y, 3, 32, 32)
                       .transpose(0, 3, 1, 4, 2)
                       .reshape(num_x * 32, num_y * 32, 3)))



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
    arr = np.zeros((samples.shape[0], samples.shape[1] * 2, samples.shape[2], samples.shape[3]))
    arr[:, 4:] = samples[:, ::-1]
    samples = np.packbits(arr.astype('int8'), axis=1)[:, 0]
    new_samples = []
    for sample in samples:
        img2 = Image.fromarray(sample)
        img2.putpalette(img.getpalette())
        img2 = img2.convert('RGB')
        new_samples.append(np.array(img2))
    samples_print = np.array(new_samples).transpose(0, 3, 1, 2)
    return samples_print

def train(num_epochs,
          filename,
          gen_lr=5e-5,
          beta_1_gen=0.5,
          beta_1_disc=0.5,
          print_freq=50,
          disc_lr=5e-5,
          num_iter_gen=1,
          n_samples=20,
          image_dir=None,
          binary_dir=None,
          gt_image_dir=None):
    
    f = h5py.File('/home/devon/Data/basic/celeba_64.hdf5', 'r')
    arr = f['features'][:1000]
    arr = arr.transpose(0, 2, 3, 1)
    arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]))
    img = Image.fromarray(arr).convert('P', palette=Image.ADAPTIVE, colors=16)

    
    # Load the dataset
    log_file = open(filename, 'w')
    print("Loading data...")
    print("Testing RW_DCGAN ...")
    log_file.write("Testing RW_DCGAN...\n")
    log_file.write("Loading data...\n")
    log_file.write("Num_epochs: {}, disc_lr: {}, gen_lr: {}\n".format(num_epochs,
                                                                      disc_lr,
                                                                      gen_lr))
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
    dim_c = input_var.shape[1]
    dim_x = input_var.shape[2]
    dim_y = input_var.shape[3]
    
    R = trng.uniform(size=(n_samples, batch_size, dim_c, dim_x, dim_y), dtype=floatX)

    g_output = lasagne.layers.get_output(generator)
    samples = (R <= T.shape_padleft(g_output)).astype(floatX)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, samples.reshape(
            (n_samples * batch_size, dim_c, dim_x, dim_y)))
    fake_out_ = fake_out.reshape((n_samples, batch_size))
    
    log_d1 = -T.nnet.softplus(-fake_out_)
    log_d0 = -(fake_out_ + T.nnet.softplus(-fake_out_))
    log_w = log_d1 - log_d0
    g_output_ = T.shape_padleft(T.clip(g_output, 1e-7, 1. - 1e-7))
    log_g = (samples * T.log(g_output_) + (1. - samples) * T.log(1. - g_output_)).sum(axis=(2, 3, 4))

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

    train_discriminator = theano.function([noise_var, input_var],
                               [(real_out > 0.).mean(), discriminator_loss],
                                allow_input_downcast=True,
                               updates=discriminator_updates)

    train_generator = theano.function([noise_var, input_var],
                               [(fake_out < 0.).mean(),
                                generator_loss, log_Z_est_.mean()],
                                allow_input_downcast=True,
                               updates=generator_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training of GAN...")
    log_file.write("Starting training of GAN...\n")
    log_file.flush()
    # We iterate over epochs:
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print("Epoch: ", epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        prefix = "ep_{}".format(epoch)
        
        for batch in train_stream.get_epoch_iterator():
            inputs = np.array(batch[0], dtype=np.float32)
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            
            train_discriminator(noise, inputs)
            disc_train_out = train_discriminator(noise, inputs)
            p_real, disc_loss = disc_train_out

            gen_loss_array = []
            p_fake_array = []
            z_est_array = []

            for i in range(num_iter_gen):
                gen_train_out = train_generator(noise, inputs)
                p_fake, gen_loss, z_est = gen_train_out
                gen_loss_array.append(gen_loss)
                p_fake_array.append(p_fake)
                z_est_array.append(z_est)

            gen_loss = np.mean(gen_loss_array)
            p_fake = np.mean(p_fake_array)
            z_est = np.mean(z_est_array)

            train_batches += 1
            if train_batches % print_freq == 0:
                print('-' * 80)
                print("Batch Number: {}, Epoch Number: {}".format(train_batches + 1, epoch + 1))
                print("Generator: p_fake: {}, gen_loss: {}, z_est: {}".format(p_fake, gen_loss, z_est))
                print("Discriminator: p_real: {}, disc_loss: {}".format(p_real, disc_loss))
                log_file.write('-' * 80 + '\n')
                log_file.write("Batch Number: {}".format(train_batches + 1, epoch + 1) + '\n')
                log_file.write("Generator: p_fake: {}, gen_loss: {} \n".format(p_fake, disc_loss))
                log_file.write("Discriminator: p_real: {}, disc_loss: {} \n".format(p_real, disc_loss))
                log_file.write('-' * 80 + '\n')
                samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
                samples = (samples >= 0.5).astype('int')
                samples = samples[0:49]
                samples_print = convert_to_rgb(samples, img)
                print_images(samples_print, 7, 7, file=image_dir + prefix + "_{}".format(train_batches) +'_gen.png')
                
                samples_print_gt = convert_to_rgb(inputs, img)
                print_images(samples_print_gt[:64], 8, 8, file=gt_image_dir + prefix + '_gt.png')

        # Then we print the results for this epoch:
        print("Total Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        log_file.write("Total Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))
        log_file.write("  training loss:\t{}\n".format(train_err / train_batches))
        log_file.flush()

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
        samples = (samples >= 0.5).astype('int')
        samples = samples[0:49]
        samples_print = convert_to_rgb(samples, img)
        print_images(samples_print, 7, 7, file=image_dir + prefix + '_gen.png')
        #if epoch == num_epochs - 1: #save binary data for further calculation
        np.savez(binary_dir + prefix + '_celeba_gen_params.npz', *lasagne.layers.get_all_param_values(generator))


    log_file.flush()
    log_file.close()



