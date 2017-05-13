'''

'''

import argparse
import datetime
import logging
import os
from os import path
import sys
import time

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import h5py
import lasagne
from lasagne.layers import (
    InputLayer, ReshapeLayer,
    DenseLayer, batch_norm, GaussianNoiseLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import pylab as pl
import theano
import theano.tensor as T
import scipy.misc
import yaml


lrelu = LeakyRectify(0.02)
floatX = theano.config.floatX
DIM_X = 64
DIM_Y = 64


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

# ##################### DATA #####################

def load_stream(batch_size=None, source=None):
    logger.info('Loading data from `{}`'.format(source))
    
    train_data = H5PYDataset(source, which_sets=('train',))
    test_data = H5PYDataset(source, which_sets=('test',))

    num_train = train_data.num_examples
    num_test = test_data.num_examples

    logger.debug('Number of test examples: {}'.format(num_test))
    logger.debug('Number of training examples: {}'.format(num_train))
    
    train_scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    train_stream = DataStream(train_data, iteration_scheme=train_scheme)
    test_scheme = ShuffledScheme(examples=num_test, batch_size=batch_size)
    test_stream = DataStream(test_data, iteration_scheme=test_scheme)
    return train_stream, num_train

def transform(image):
    return np.array(image) / 127.5 - 1.  # seems like normalization

def inverse_transform(image):
    return (np.array(image) + 1.) * 127.5

def print_images(images, num_x, num_y, file='./'):
    scipy.misc.imsave(file,  # current epoch No.
                      (images.reshape(num_x, num_y, 3, DIM_X, DIM_Y)
                       .transpose(0, 3, 1, 4, 2)
                       .reshape(num_x * DIM_X, num_y * DIM_Y, 3)))

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

def build_discriminator(input_var=None, dim_h=128, **kwargs):
    layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    layer = Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = batch_norm(Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, dim_h * 4, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, dim_h * 8, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    logger.debug('Discriminator output: {}' .format(layer.output_shape))
    return layer

def build_generator(input_var=None, dim_z=100, dim_h=128, **kwargs):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import tanh

    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, dim_h * 8 * 4 * 4))
    layer = ReshapeLayer(layer, ([0], dim_h * 8, 4, 4))
    layer = batch_norm(Deconv2DLayer(layer, dim_h * 4, 5, stride=2, pad=2))
    layer = batch_norm(Deconv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 3, 5, stride=2, pad=2, nonlinearity=tanh)
    
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

def norm_exp(log_factor):
    '''Gets normalized weights.

    '''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde


def est_log_Z(fake_out):
    log_w = fake_out
    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    return log_Z_est

# ##################### LOSSES #######################

def BGAN(fake_out, real_out, log_Z=None, use_log_Z=True):
    '''Nonlinearity of discriminator is sigmoid.
    
    '''
    if use_log_Z:
        generator_loss = ((fake_out - log_Z) ** 2).mean()
    else:
        generator_loss = (fake_out ** 2).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    return generator_loss, discriminator_loss


def LSGAN(fake_out, real_out, target=1.0):
    generator_loss = ((fake_out - target) ** 2).mean()
    discriminator_loss = ((real_out - 1.) ** 2).mean() + (fake_out ** 2).mean()
    return generator_loss, discriminator_loss


def WGAN(fake_out, real_out):    
    generator_loss = fake_out.mean()
    discriminator_loss = real_out.mean() - fake_out.mean()
    return generator_loss, discriminator_loss


def GAN(fake_out, real_out):
    generator_loss = T.nnet.softplus(-fake_out).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    return generator_loss, discriminator_loss

# ##################### MAIN #######################

def main(data_args, optimizer_args, model_args, train_args,
         image_dir=None, binary_dir=None,
         summary_updates=200, debug=False):
      
    # DATA  
    train_stream, training_samples = load_stream(**data_args)
    
    # VAR
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')

    # MODELS
    logger.info('Building model and compiling GAN functions...')
    generator = build_generator(noise_var, **model_args)
    discriminator = build_discriminator(input_var, **model_args)

    # GRAPH / LOSS
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))

    log_Z_est = None
    loss = model_args['loss']
    loss_args = model_args.get('loss_args', {})
    if loss == 'bgan':
        logger.info('Using BGAN')
        generator_loss, discriminator_loss = BGAN(
            fake_out, real_out, log_Z, **loss_args)
        log_Z_est = est_log_Z(fake_out)
    elif loss == 'gan':
        logger.info('Using normal GAN')
        generator_loss, discriminator_loss = GAN(fake_out, real_out)
        log_Z_est = est_log_Z(fake_out)
    elif loss == 'lsgan':
        logger.info('Using LSGAN')
        generator_loss, discriminator_loss = LSGAN(fake_out, real_out)
    elif loss == 'wgan':
        logger.info('Using WGAN')
        generator_loss, discriminator_loss = WGAN(fake_out, real_out)
        
    # OPTIMIZER
    generator_params = lasagne.layers.get_all_params(
        generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(
        discriminator, trainable=True)

    generator_updates = lasagne.updates.adam(
        generator_loss, generator_params, **optimizer_args)
    discriminator_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, **optimizer_args)
    if loss == 'wgan':
        for k in discriminator_updates.keys():
            if k.name == 'W':
                discriminator_updates[k] = T.clip(
                    discriminator_updates[k], -0.01, 0.01)
    
    if log_Z_est is not None:
        generator_updates.update(
            [(log_Z, 0.995 * log_Z + 0.005 * log_Z_est.mean())])

    d_results = {
        'p(real)': (real_out > 0.).mean(),
        'L_D': discriminator_loss
    }
    
    g_results = {
        'p(fake)': (fake_out < 0.).mean(),
        'L_G': generator_loss
    }
    if log_Z_est is not None:
        g_results.update(**{
            'log Z': log_Z,
            'log Z (est)': log_Z_est.mean()
        })

    # COMPILE
    train_discriminator = theano.function([noise_var, input_var],
        d_results, allow_input_downcast=True, updates=discriminator_updates)

    train_generator = theano.function([noise_var],
        g_results, allow_input_downcast=True, updates=generator_updates)

    gen_fn = theano.function(
        [noise_var], lasagne.layers.get_output(generator, deterministic=True))

    # TRAIN
    logger.info('Starting training of GAN...')

    for epoch in range(train_args['epochs']):
        logger.info('Epoch: '.format(epoch))
        u = 0
        start_time = time.time()
        prefix = 'ep_{}'.format(epoch)
        
        results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets,
            maxval=(training_samples // data_args['batch_size'])).start()
        
        for batch in train_stream.get_epoch_iterator():
            inputs = transform(np.array(batch[0], dtype=np.float32))
            if inputs.shape[0] == data_args['batch_size']:
                noise = lasagne.utils.floatX(
                    np.random.rand(len(inputs), model_args['dim_z']))
    
                for i in range(train_args['num_iter_disc']):
                    d_outs = train_discriminator(noise, inputs)
                    d_outs =  dict((k, np.asarray(v)) for k, v in d_outs.items())
                    update_dict_of_lists(results, **d_outs)
    
                for i in range(train_args['num_iter_gen']):
                    g_outs = train_generator(noise)
                    g_outs = dict((k, np.asarray(v)) for k, v in g_outs.items())
                    update_dict_of_lists(results, **g_outs)
    
                u += 1
                pbar.update(u)
                if summary_updates is not None and u % summary_updates == 0:
                    result_summary = dict((k, np.mean(v)) for k, v in results.items())
                    logger.info(result_summary)
                    
                    samples = gen_fn(lasagne.utils.floatX(np.random.rand(
                        5000, model_args['dim_z'])))
                    samples_print = samples[0:64]
                    print_images(inverse_transform(samples_print), 8, 8,
                                 file=path.join(image_dir, prefix + '_gen_tmp.png'))

        logger.info('Total Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, train_args['epochs'], time.time() - start_time))
        
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(
            5000, model_args['dim_z'])))
        samples_print = samples[0:64]
        print_images(inverse_transform(samples_print), 8, 8,
                     file=path.join(image_dir, prefix + '_gen.png'))
        np.savez(path.join(binary_dir, prefix + '_celeba_gen_params.npz'),
                 *lasagne.layers.get_all_param_values(generator))


    log_file.flush()
    log_file.close()


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
    parser.add_argument('-c', '--config_file', default=None)
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
    if not path.isdir(binary_dir):
        os.mkdir(binary_dir)
    if not path.isdir(image_dir):
        os.mkdir(image_dir)
    if not path.isdir(gt_image_dir):
        os.mkdir(gt_image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir)


def config(data_args, model_args, optimizer_args, train_args,
           config_file=None):
    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        model_args.update(**d.get('model_args', {}))
        optimizer_args.update(**d.get('optimizer_args', {}))
        train_args.update(**d.get('train_args', {}))
        data_args.update(**d.get('data_args', {}))


_default_data_args = dict(
    batch_size=64
)

_default_optimizer_args = dict(
    learning_rate=1e-3,
    beta1=0.5
)

_default_model_args = dict(
    dim_z=64,
    dim_h=128,
    loss='bgan',
    loss_args=dict(use_log_Z=True)
)

_default_train_args = dict(
    epochs=100,
    num_iter_gen=1,
    num_iter_disc=1,
)


if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    kwargs = dict()
    kwargs.update(out_paths)
    logger.info('kwargs: {}'.format(kwargs))
    
    data_args = {}
    data_args.update(**_default_data_args)
    data_args['source'] = args.source
    
    optimizer_args = {}
    optimizer_args.update(_default_optimizer_args)
    
    model_args = {}
    model_args.update(**_default_model_args)
    
    train_args = {}
    train_args.update(**_default_train_args)
    config(data_args, model_args, optimizer_args, train_args,
           config_file=args.config_file)
    
    logger.info('Training model with: \ndata args {}, \noptimizer args {} '
                '\nmodel args {} \ntrain args {}'.format(
                    data_args, optimizer_args, model_args, train_args))
    
    main(data_args, optimizer_args, model_args, train_args, **kwargs)  
