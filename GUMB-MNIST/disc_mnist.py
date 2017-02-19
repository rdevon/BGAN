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


def build_generator(input_var=None, estimator="ST", temperature=None):
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
    print("Using estimator: ", estimator)
    if estimator=="GS":
        layer = GumbelSoftmaxLayer(layer, K=28, hard=False, temperature=temperature)
    elif estimator=="GS-ST":
        layer = GumbelSoftmaxLayer(layer, K=28, hard=True, temperature=temperature)
    else:
        layer = STLayer(layer, K=28)

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
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print ("Discriminator output:", layer.output_shape)
    return layer

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=40, initial_eta=2e-4, temp_init=3.0):
    anneal_interval = 300
    anneal_rate = 0.001
    # Load the dataset
    print("Loading data...")
    source = "/u/jacobath/cortex-data/basic/mnist_binarized_salakhutdinov.pkl.gz"
    X_train= load_dataset(source=source, mode="train")

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    temperature = T.scalar("Temp")
    tau = temp_init

    # Create neural network model
    print("Building model and compiling functions...")

    generator = build_generator(noise_var, temperature=temperature)
    discriminator = build_discriminator(input_var)

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
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=eta*0.001, beta1=0.5)
    updates.update(lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=eta*0.1, beta1=0.5))

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
    gen_fn = theano.function([noise_var, temperature],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True),
                               allow_input_downcast=True,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")
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
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))
        # And finally, we plot some generated data
        prefix = "%d_" % epoch
        if epoch % 1 == 0:
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(100, 100)), tau)
            import matplotlib.pyplot as plt
            plt.imsave('./gen_images/' + prefix + '.png',
                       (samples.reshape(10, 10, 28, 28)
                        .transpose(0, 2, 1, 3)
                        .reshape(10 * 28, 10 * 28)),
                       cmap='gray')



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        main()
