#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

from data_reader import DataGenerator
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib
floatX = theano.config.floatX
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from lasagne.layers import (InputLayer, DenseLayer)
from lasagne.nonlinearities import rectify, tanh

def normalize(x):
    return tanh(x/4.0)

def build_discriminator(input_var=None, batch_size=512):
    layer = InputLayer(shape=(batch_size, 2), input_var=input_var)
    layer = DenseLayer(layer, num_units=128, nonlinearity=tanh)
    layer = DenseLayer(layer, num_units=1, nonlinearity=None)
    print("Discriminator output:", layer.output_shape)
    return layer

def build_generator(input_var=None, batch_size=512):
    from lasagne.layers import InputLayer, DenseLayer
    parameters = {}
    layer = InputLayer(shape=(batch_size, 256), input_var=input_var)
    layer = DenseLayer(layer, num_units=128, nonlinearity=tanh, W=lasagne.init.Orthogonal(0.8))
    parameters['W2'] = layer.W
    parameters['b2'] = layer.b
    layer = DenseLayer(layer, num_units=128, nonlinearity=tanh, W=lasagne.init.Orthogonal(0.8))
    parameters['W3'] = layer.W
    parameters['b3'] = layer.b
    layer = DenseLayer(layer, num_units=2, nonlinearity=None)
    parameters['W4'] = layer.W
    parameters['b4'] = layer.b
    print("Fake generator output:", layer.output_shape)
    return parameters, layer


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

def train(num_epochs,
          filename,
          batch_size = 512,
          num_elements = 300,
          disc_lr=0.01,
          gen_lr=0.01,
          image_dir='./save/',
          ):

    # Load the dataset
    log_file = open(filename, 'w')
    print("Loading data...")
    print("Testing DCGAN + modified Loss...")
    log_file.write("Testing DCGAN + modified Loss...\n")
    log_file.write("Loading data...\n")
    log_file.write("Num_epochs: {}, disc_lr: {}, gen_lr: {}\n".format(num_epochs, disc_lr, gen_lr))
    log_file.flush()

    # if setting=='d31':
    #     batch_size = 3100
    #     num_points = 3100
    # elif setting=='r15':
    #     batch_size = 1000
    #     num_points = 1000
    # elif setting=='spiral':
    #     batch_size = 624
    #     num_points = 624
    # else:
    #     raise Exception

    # Prepare Theano variables for inputs and targets
    noise_var = T.fmatrix('noise')
    input_var = T.fmatrix('inputs - test')

    # Create neural network model
    print("Building model and compiling GAN functions...")
    log_file.write("Building model and compiling GAN functions...\n")
    _, generator = build_generator(noise_var, batch_size)
    discriminator = build_discriminator(input_var, batch_size)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))

    generator_loss = 0.5*reweighted_loss(fake_out)

    # Create disc_loss
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (T.nnet.softplus(-fake_out)).mean() + fake_out.mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    #Generator loss
    generator_updates = lasagne.updates.rmsprop(
        generator_loss, generator_params, learning_rate=gen_lr)

    #Discriminator loss
    discriminator_updates = lasagne.updates.rmsprop(
        discriminator_loss, discriminator_params, learning_rate=disc_lr)


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
                                                       deterministic=True),
                             allow_input_downcast=True,
                             on_unused_input='ignore')


    # Finally, launch the training loop.
    data_iter = DataGenerator(image_dir)
    data_iter.load_training_data(batch_size=batch_size, num_elements=num_elements)
    print("Starting training of GAN...")
    log_file.write("Starting training of GAN...\n")
    print("Data_iter Num batches: ", data_iter.num_batches)
    log_file.flush()
    num_batches = data_iter.num_batches
    cov = np.identity(256)
    mean = np.zeros(256)

    for epoch in range(num_epochs):
        print("Epochs: ", epoch)
        data_iter.reset()

        for batch_num in range(num_batches):
            # In each epoch, we do a full pass over the training data:
            X = data_iter.get_batch()
            noise = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)

            # for i in range(9):
            train_discriminator(noise, X)
            disc_train_out = train_discriminator(noise, X)
            gen_train_out = train_generator(noise)
            p_real, disc_loss = disc_train_out
            p_fake, gen_loss = gen_train_out
            print('-'*80)
            print("Batch Number: {}, Epoch Number: {}".format(batch_num+1, epoch+1))
            print("Generator: p_fake: {}, gen_loss: {}".format(p_fake, disc_loss))
            print("Discriminator: p_real: {}, disc_loss: {}".format(p_real, disc_loss))
            log_file.write('-' * 80 + '\n')
            log_file.write("Batch Number: {}".format(batch_num+1, epoch+1) + '\n')
            log_file.write("Generator: p_fake: {}, gen_loss: {} \n".format(p_fake, disc_loss))
            log_file.write("Discriminator: p_real: {}, disc_loss: {} \n".format(p_real, disc_loss))
            log_file.write('-' * 80 + '\n')
            log_file.flush()
            # And finally, we plot some generated data
            if (batch_num+1)%5==0:
                samples = gen_fn(np.random.multivariate_normal(mean=mean, cov=cov, size=1000))
                xval = samples[:,0]
                yval = samples[:,1]
                plt.title("Figure plot")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                plt.axis([-10.0, 10.0, -10.0, 10.0])
                plt.scatter(xval, yval)
                plt.grid()
                plt.legend()
                plt.draw()
                plt.savefig(image_dir + '/plot_batch_{}_epoch_{}'.format(batch_num+1, epoch+1))
                plt.clf()
                plt.cla()
                plt.close()

    log_file.flush()
    log_file.close()
