from __future__ import print_function

import random
import sys
import os
import time

import numpy as np
import theano
import lasagne.layers as L
import theano.tensor as T
import pylab as pl
import lasagne
import scipy.misc
import cPickle
import gzip
from rw_mnist import build_generator
from lasagne.layers import (InputLayer, ReshapeLayer,
                            DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
floatX = theano.config.floatX

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def generate_samples():
    noise_var = T.matrix('noise')
    # Create neural network model

    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    output = L.get_output(generator, deterministic=True)
    print("Loading parameters...")
    generator_params = np.load("./gen_binaries/epoch_190_lr_0.0001_disc_mnist_gen_params.npz")
    generator_params = [generator_params['arr_%d' % i] for i in range(len(generator_params.files))]
    L.set_all_param_values(generator, generator_params)

    print("Compiling functions....")
    gen_fn = theano.function([noise_var], output)
    print("Starting generations....")

    array = [[3, 7, 54, 40, 46, 179, 188, 200, 157, 168],
                [22, 23, 25, 10, 12, 14, 110, 8, 85, 89],
                [67, 65, 115, 350, 442, 491, 543, 587, 766, 781],
             [0, 9, 20, 43, 62, 64, 131, 177, 190, 222],
             [38, 96, 113, 118, 119, 174, 248, 259, 351, 342],
             [66, 78, 99, 117, 128, 208, 280, 560, 991, 978],
             [53, 162, 199, 229, 224, 289, 319, 317, 440, 472],
             [212, 279, 278, 290, 299, 336, 337, 371, 380, 421],
             [235, 400, 420, 450, 459, 471, 482, 544, 616, 68],
             [198, 332, 460, 484, 496, 497, 503, 535, 537, 537, 588]]

    samples = np.ones((100, 1, 28, 28))
    counter = 0
    for i in range(10):
        subarray = array[i]
        np.random.seed(1234)
        for i in range(1000):
            print("Index: ", i)

            input_noise = lasagne.utils.floatX(np.random.rand(1, 100))
            sample = gen_fn(input_noise)
            if i in subarray:
                samples[counter] = sample
                counter+=1

    import matplotlib.pyplot as plt
    plt.imsave('./generations/' + "digits" + '_.png',
               (samples.reshape(10, 10, 28, 28)
               .transpose(0, 2, 1, 3)
               .reshape(10 * 28, 10 * 28))
               , cmap='gray')


generate_samples()