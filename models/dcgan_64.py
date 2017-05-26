'''Standard DCGAN model for 64x64 images

'''

import logging

from lasagne.layers import (
    batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer,
    TransposedConv2DLayer)
from lasagne.nonlinearities import LeakyRectify, tanh


logger = logging.getLogger('BGAN.models.dcgan_64')

DIM_X = 64
DIM_Y = 64
DIM_C = 3
NONLIN = tanh


def build_discriminator(input_var=None, dim_h=None, use_batch_norm=True,
                        leak=None):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)

    layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
    layer = Conv2DLayer(layer, dim_h, 4, stride=2, pad=1, nonlinearity=lrelu)
    logger.debug('Discriminator output 1: {}' .format(layer.output_shape))
    layer = bn(Conv2DLayer(layer, dim_h * 2, 4, stride=2, pad=1,
                           nonlinearity=lrelu))
    logger.debug('Discriminator output 2: {}' .format(layer.output_shape))
    layer = bn(Conv2DLayer(layer, dim_h * 4, 4, stride=2, pad=1,
                           nonlinearity=lrelu))
    logger.debug('Discriminator output 3: {}' .format(layer.output_shape))
    layer = bn(Conv2DLayer(layer, dim_h * 8, 4, stride=2, pad=1,
                           nonlinearity=lrelu))
    logger.debug('Discriminator output 4: {}' .format(layer.output_shape))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    
    logger.debug('Discriminator output: {}' .format(layer.output_shape))
    return layer


def build_generator(input_var=None, dim_z=None, dim_h=None):
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    
    layer = batch_norm(DenseLayer(layer, dim_h * 8 * 4 * 4))
    layer = ReshapeLayer(layer, ([0], dim_h * 8, 4, 4))
    logger.debug('Generator output 1: {}' .format(layer.output_shape))
    layer = batch_norm(TransposedConv2DLayer(
        layer, dim_h * 4, 4, stride=2, crop=1))
    logger.debug('Generator output 2: {}' .format(layer.output_shape))
    layer = batch_norm(TransposedConv2DLayer(
        layer, dim_h * 2, 4, stride=2, crop=1))
    logger.debug('Generator output 3: {}' .format(layer.output_shape))
    layer = batch_norm(TransposedConv2DLayer(
        layer, dim_h, 4, stride=2, crop=1))
    logger.debug('Generator output 4: {}' .format(layer.output_shape))
    layer = TransposedConv2DLayer(
        layer, DIM_C, 4, stride=2, crop=1, nonlinearity=NONLIN)
    
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer
