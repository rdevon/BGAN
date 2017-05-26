'''DCGAN for 28x28 images. Used for MNIST. Published version.

'''

import logging

from lasagne.layers import (
    batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer)
from lasagne.nonlinearities import LeakyRectify, tanh

from deconv import Deconv2DLayer


logger = logging.getLogger('BGAN.models.dcgan_28_pub')

DIM_X = 28
DIM_Y = 28
DIM_C = 1
NONLIN = None


def build_generator(input_var=None, dim_z=100, dim_h=64):
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, dim_h * 2 * 7 * 7))
    layer = ReshapeLayer(layer, ([0], dim_h * 2, 7, 7))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, DIM_C, 5, stride=2, pad=2,
                          nonlinearity=None)
    
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer


def build_discriminator(input_var=None, dim_h=64, use_batch_norm=True,
                        leak=None):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)
    
    layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
    
    layer = bn(Conv2DLayer(
        layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = bn(Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2,
                           nonlinearity=lrelu))
    layer = DenseLayer(layer, 1024, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=NONLIN)
    
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer