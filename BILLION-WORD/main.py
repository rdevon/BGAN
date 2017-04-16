'''Module for training BGAN on Billion Word

'''

import datetime
import logging
import sys
import time
import os

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import OneHotEncoding
import h5py
import lasagne
from lasagne.layers import (
    batch_norm, Conv1DLayer, DenseLayer, GaussianNoiseLayer, InputLayer,
    ReshapeLayer)
from lasagne.nonlinearities import LeakyRectify, sigmoid, softmax
import numpy as np
from PIL import Image
from progressbar import Bar, ProgressBar, Percentage, Timer
import random
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc


floatX = theano.config.floatX
lrelu = LeakyRectify(0.2)

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

def load_stream(batch_size=64, source=None):
    if source is None:
        raise ValueError('Source not set.')
    data = np.load(source)
    f = h5py.File('dataset.hdf5', mode='w')
    
    image_features = f.create_dataset(
        'features', data.shape, dtype=data.dtype)
    image_features[...] = data
    
    train_scheme = ShuffledScheme(examples=data.shape[0], batch_size=batch_size)
    train_stream = OneHotEncoding(data_stream=DataStream(
        data, iteration_scheme=train_scheme), num_classes=N_WORDS)
    return train_stream
            
# ##################### MODEL #####################

def build_generator(parameter, input_var=None):
    layer = InputLayer(shape=(None, 100), input_var=input_var)

    # fully-connected layer
    layer = DenseLayer(layer, 256 * 128)
    layer = ReshapeLayer(layer, ([0], 64, 256))
    layer = batch_norm(Conv1DLayer(layer, 32, 5, stride=2, pad=2))
    layer = batch_norm(Conv1DLayer(layer, 16, 5, stride=2, pad=2))
    layer = batch_norm(Conv1DLayer(layer, N_WORDS, 5, stride=2, pad=2,
                                   nonlinearity=None))
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def build_discriminator(input_var=None):
    layer = InputLayer(shape=(None, N_WORDS, 32), input_var=input_var)
    layer = Conv1DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv1DLayer(layer, 128 * 2, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = Conv1DLayer(layer, 128 * 4, 5, stride=2, pad=2, nonlinearity=lrelu)
    layer = DenseLayer(layer, 1, nonlinearity=None)
    
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer

# ##################### MATH #####################

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

# ##################### LOSS #####################

def BGAN(discriminator, g_output_logit, n_samples):
    g_output_logit_ = g_output_logit.transpose(0, 2, 1)
    g_output_logit_ = g_output_logit_.reshape((-1, N_WORDS))
    
    g_output = T.nnet.softmax(g_output_logit_)
    g_output = g_output.reshape((-1, L_GEN, N_WORDS))
    
    p_p = T.shape_padleft(g_output)
    g_output = g_output.transpose(0, 2, 1)
    p_t = T.zeros((n_samples, batch_size, L_GEN, N_WORDS)) + p_p
    p = p_t.reshape((-1, N_WORDS))
    
    samples = trng.multinomial(pvals=p).astype(floatX)
    samples = theano.gradient.disconnected_grad(samples)
    samples = samples.reshape((n_samples, -1, L_GEN, N_WORDS))
    samples = samples.transpose(0, 1, 3, 2)
    
    D_r = lasagne.layers.get_output(discriminator)
    D_f = lasagne.layers.get_output(
        discriminator, samples.reshape((-1, N_WORDS, L_GEN)))
    D_f_ = D_f.reshape((n_samples, -1))
    
    log_d1 = -T.nnet.softplus(-fake_out_)
    log_d0 = -(D_f_ + T.nnet.softplus(-fake_out_))
    log_w = fake_out_

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    
    log_g = (samples * (g_output_logit - log_sum_exp2(
        g_output_logit, axis=1))[None, :, :, :]).sum(axis=(2, 3))
         
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
    
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (
        T.nnet.softplus(-fake_out)).mean()
    
    return generator_loss, discriminator_loss, D_r, D_f, log_Z_est

# MAIN -------------------------------------------------------------------------

def main(source=None, num_epochs=None, gen_lr=None, beta_1_gen=None,
         dim_noise=None, initial_eta=None, batch_size=None,
         beta_1_disc=None, disc_lr=None, num_iter_gen=None, n_steps=None,
         print_freq=None, image_dir=None, binary_dir=None, gt_image_dir=None):
    
    # Load the dataset
    data = np.load(source)
    train_samples = data.shape[0]
    
    # VAR
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    
    # MODELS
    generator = build_generator(parameter, noise_var)
    discriminator = build_discriminator(input_var)

    # GRAPH / LOSS    
    g_output_logit = lasagne.layers.get_output(generator)
    
    generator_loss, discriminator_loss, D_r, D_f, log_Z_est = BGAN(
        discriminator, g_output_logit, n_samples)
    
    # OPTIMIZER
    discriminator_params = lasagne.layers.get_all_params(
        discriminator, trainable=True)
    generator_params = lasagne.layers.get_all_params(
        generator, trainable=True)
    
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    
    l_kwargs = dict(learning_rate=eta, beta1=0.5)
    
    updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, **l_kwargs)
    updates.update(lasagne.updates.adam(
        generator_loss, generator_params, **l_kwargs))
    
    outputs = {
        'G cost': generator_loss,
        'D cost': discriminator_loss,
        'p(real)': T.nnet.sigmoid(D_r > .5).mean(),
        'p(fake)': T.nnet.sigmoid(D_f > .5).mean(),
    }
    train_fn = theano.function([input_var, noise], outputs, updates=updates)
    gen_fn = theano.function([noise], lasagne.layers.get_output(
        generator, deterministic=True))

    # train
    logger.info('Starting training...')
    
    for epoch in range(num_epochs):
        train_batches = 0
        start_time = time.time()
        
        # Train
        results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        
        for batch in train_stream.get_epoch_iterator():
            if batch0 is None: batch0 = batch
            noise = lasagne.utils.floatX(np.random.rand(batch[0].shape[0],
                                                        dim_noise))
            outs = train_fn(*(list(batch)[::-1] + [noise]))
            update_dict_of_lists(results, **outs)
            train_batches += 1
            pbar.update(train_batches)
        
        # Summarize and print
        results = dict((k, np.mean(v)) for k, v in results.items())

        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info(results)

        # Plot
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(100, dim_noise)))
        assert False, samples.shape

        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta * 2 * (1 - progress)))
    
_defaults = dict(
    gen_lr=5e-5,
    beta_1_gen=0.5,
    beta_1_disc=0.5,
    disc_lr=5e-5,
    num_epochs=100,
    num_iter_gen=1,
    dim_noise=64,
    batch_size=64,
    n_steps=2,
    initial_eta=1e-4,
    print_freq=50
)
    
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
        os.mkdir(binary_dir)
        os.mkdir(image_dir)
        os.mkdir(gt_image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir, gt_image_dir=gt_image_dir)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    kwargs = dict()
    kwargs.update(**_defaults)
    kwargs.update(out_paths)
        
    main(source=args.source, **kwargs)