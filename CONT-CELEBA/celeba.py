#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import datetime
import sys
import time
import os
import numpy as np
import theano
import theano.tensor as T
import pylab as pl
import lasagne
import scipy.misc
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from lasagne.layers import (InputLayer, ReshapeLayer,
                                DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
floatX = theano.config.floatX
data_name = "celeba_64.hdf5"


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def load_stream(batch_size=64, path="/data/lisa/data"):
    path = os.path.join(path, data_name)
    train_data = H5PYDataset(path, which_sets=('train',))
    test_data = H5PYDataset(path, which_sets=('test',))
    num_train = train_data.num_examples
    num_test = test_data.num_examples
    print("Number of test examples: ", num_test)
    print("Number of training examples: ", num_train)
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
    layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    # two convolutions
    layer = Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = batch_norm(Conv2DLayer(layer, 128 * 2, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 4, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128 * 8, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print("Discriminator output:", layer.output_shape)
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

    # fully-connected layer
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
    # shape=(batch,1,28,28)
    print("Generator output:", layer.output_shape)
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

def reweighted_loss(fake_out):
    log_d1 = -T.nnet.softplus(-fake_out)  # -D_cell.neg_log_prob(1., P=d)
    log_d0 = -(fake_out+T.nnet.softplus(-fake_out))  # -D_cell.neg_log_prob(0., P=d)
    log_w = log_d1 - log_d0

    # Find normalized weights.
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N

    #cost = (log_w ** 2).mean()
    #cost = ((log_w - T.maximum(log_Z_est, -2)) ** 2).mean()
    cost = (log_Z_est ** 2).mean()
    
    return cost


def train(num_epochs,
          filename,
          gen_lr=1e-5,
          beta_1_gen=0.5,
          beta_1_disc=0.5,
          print_freq=200,
          disc_lr=1e-5,
          num_iter_gen=1,
          image_dir=None,
          binary_dir=None):

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
    train_stream, test_stream = load_stream()
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

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))


    generator_loss = 0.5 * reweighted_loss(fake_out)
    #generator_loss = (T.nnet.softplus(-fake_out)).mean() -- Original GAN loss

    # Create disc_loss
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (T.nnet.softplus(-fake_out)).mean() + fake_out.mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    # Generator loss
    generator_updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=gen_lr, beta1=beta_1_gen)

    # Discriminator loss
    discriminator_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=disc_lr, beta1=beta_1_disc)


    train_discriminator = theano.function([noise_var, input_var],
                               [(real_out > 0.).mean(), discriminator_loss],
                                allow_input_downcast=True,
                               updates=discriminator_updates)

    train_generator = theano.function([noise_var],
                               [(fake_out < 0.).mean(),
                                generator_loss],
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
            inputs = transform(np.array(batch[0],dtype=np.float32))  # or batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))

            train_discriminator(noise, inputs)
            disc_train_out = train_discriminator(noise, inputs)
            p_real, disc_loss = disc_train_out

            gen_loss_array = []
            p_fake_array = []

            for i in range(num_iter_gen):
                gen_train_out = train_generator(noise)
                p_fake, gen_loss = gen_train_out
                gen_loss_array.append(gen_loss)
                p_fake_array.append(p_fake)

            gen_loss = np.mean(gen_loss_array)
            p_fake = np.mean(p_fake_array)

            train_batches += 1
            if train_batches % print_freq == 0:
                print('-' * 80)
                print("Batch Number: {}, Epoch Number: {}".format(train_batches + 1, epoch + 1))
                print("Generator: p_fake: {}, gen_loss: {}".format(p_fake, gen_loss))
                print("Discriminator: p_real: {}, disc_loss: {}".format(p_real, disc_loss))
                log_file.write('-' * 80 + '\n')
                log_file.write("Batch Number: {}".format(train_batches + 1, epoch + 1) + '\n')
                log_file.write("Generator: p_fake: {}, gen_loss: {} \n".format(p_fake, disc_loss))
                log_file.write("Discriminator: p_real: {}, disc_loss: {} \n".format(p_real, disc_loss))
                log_file.write('-' * 80 + '\n')
                samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
                samples_print = samples[0:49]
                print_images(inverse_transform(samples_print), 7, 7, file=image_dir + prefix + '_gen_tmp.png')

        # Then we print the results for this epoch:
        print("Total Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        log_file.write("Total Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time))

        log_file.write("  training loss:\t{}\n".format(train_err / train_batches))
        log_file.flush()

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(5000, 100)))
        samples_print = samples[0:49]
        print_images(inverse_transform(samples_print), 7, 7, file=image_dir + prefix + '_gen.png')
        #if epoch == num_epochs - 1: #save binary data for further calculation
        np.savez(binary_dir + prefix + '_celeba_gen_params.npz', *lasagne.layers.get_all_param_values(generator))


    log_file.flush()
    log_file.close()



