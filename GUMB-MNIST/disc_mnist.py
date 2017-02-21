#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Deep Convolutional Generative Adversarial Networks
(DCGANs, see http://arxiv.org/abs/1511.06434).
It is based on the MNIST example in Lasagne:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
Note: In contrast to the original paper, this trains the generator and
discriminator at once, not alternatingly. It's easy to change, though.
Jan Schlüter, 2015-12-16
"""
from __future__ import print_function

import sys
import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import time

import numpy as np
import theano
import theano.tensor as T
import pylab as pl
import lasagne
import scipy.misc
import cPickle
import gzip
from lasagne.layers import (InputLayer, ReshapeLayer,
                            DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
from layers.gs_layer import GumbelSoftmaxLayer
from layers.st_layer import STLayer
floatX = theano.config.floatX
log_file = None
#Dataset loader
def load_dataset(source, mode):
    print("Reading MNIST, ", mode)
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

#Dataset iterator
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


# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

#Transponsed/fractional-strided convolutional layer
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


def build_generator(input_var=None, hard=True, temperature=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2,
                          nonlinearity=sigmoid)

    layer = GumbelSoftmaxLayer(layer, K=28, hard=hard, temperature=temperature)

    print("Generator output:", layer.output_shape)
    return layer

def build_discriminator(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm)
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu)
    # fully-connected layer
    layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print ("Discriminator output:", layer.output_shape)
    return layer

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.


def train(gumbel_hard, optimGD, lr, anneal_rate, anneal_interval, num_epochs=20, temp_init=3.0):
    prefix = "{}_{}_{}_{}_{}".format(gumbel_hard,
                        optimGD,
                        lr,
                        anneal_rate,
                        anneal_interval)
    global log_file
    log_file = open("./gen_logs/%s_log.txt" % prefix, 'w')

    # Load the dataset
    log_file.write("Loading data...\n")
    #source = "/u/jacobath/cortex-data/basic/mnist_binarized_salakhutdinov.pkl.gz"
    source = "/home/apjacob/data/mnist_binarized_salakhutdinov.pkl.gz"
    X_train= load_dataset(source=source, mode="train")

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    temperature = T.scalar("Temp")
    tau = temp_init

    # Create neural network model
    log_file.write("Building model and compiling functions...\n")

    generator = build_generator(noise_var, temperature=temperature, hard=gumbel_hard)
    discriminator = build_discriminator(input_var)
    log_file.write("Generator and discimininator built...\n")
    log_file.flush()
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))


    #Create gen_loss
    generator_loss = (T.nnet.softplus(-fake_out)).mean()

    #Create disc_loss
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (T.nnet.softplus(-fake_out)).mean() + fake_out.mean()

    #Get generator and discriminator params
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    #Set optimizers and learning rates
    if optimGD=='adam':
        updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=lr, beta1=0.5)
        updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5))
    elif optimGD=='sgd':
        updates = lasagne.updates.sgd(
            generator_loss, generator_params, learning_rate=lr)
        updates.update(lasagne.updates.sgd(
            discriminator_loss, discriminator_params, learning_rate=lr))
    else:
        updates = lasagne.updates.rmsprop(
            generator_loss, generator_params, learning_rate=lr)
        updates.update(lasagne.updates.rmsprop(
            discriminator_loss, discriminator_params, learning_rate=lr))

    log_file.write("Compiling train function...\n")
    log_file.flush()
    # Compile a training function
    train_fn = theano.function([noise_var, input_var, temperature],
                               [(real_out > 0.).mean(),
                                (fake_out < 0.).mean(),
                                generator_loss,
                                discriminator_loss,
                                temperature],
                               updates=updates,
                               allow_input_downcast=True,
                               on_unused_input='ignore')

    # Compile another generating function
    log_file.write("Compiling generation function...\n")
    log_file.flush()

    gen_fn = theano.function([noise_var, temperature],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True),
                               allow_input_downcast=True,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    log_file.write("Starting training...\n")
    log_file.flush()
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        counter = 0
        for batch in iterate_minibatches(X_train, 128, shuffle=True):
            inputs = np.array(batch, dtype=np.float32)
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            train_err += np.array(train_fn(noise, inputs, tau))
            train_batches += 1
            counter += 1
            if counter % anneal_interval == 0:
                tau = np.maximum(tau * np.exp(-anneal_rate * counter), 0.5)
        # Then we print the results for this epoch:
        log_file.write("Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time))
        log_file.write("  training loss:\t\t{}\n".format(train_err / train_batches))
        log_file.flush()

        if epoch%3==0:
            log_file.write("Generating Samples...\n")
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(100, 100)), tau)
            import matplotlib.pyplot as plt
            plt.imsave('./gen_images/' + prefix + "_epoch_{}".format(epoch) + '.png',
                       (samples.reshape(10, 10, 28, 28)
                        .transpose(0, 2, 1, 3)
                        .reshape(10 * 28, 10 * 28)),
                       cmap='gray')


if __name__ == '__main__':
    gumbel_hard = bool(sys.argv[1])
    optimGD = sys.argv[2]
    learning_rate = float(sys.argv[3])
    anneal_rate = float(sys.argv[4])
    anneal_interval = int(sys.argv[5])
    train(gumbel_hard, optimGD, learning_rate, anneal_rate, anneal_interval)
    log_file.close()

