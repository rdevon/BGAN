'''Module for training BGAN on Billion Word

'''

import argparse
import cPickle as pickle
import datetime
import logging
import os
from os import path
import sys
import time

from collections import OrderedDict
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import SourcewiseTransformer
import h5py
import lasagne
from lasagne.layers import (
    batch_norm, Conv1DLayer, DenseLayer, ElemwiseSumLayer,
    GaussianNoiseLayer, InputLayer, NonlinearityLayer, ReshapeLayer)
from lasagne.nonlinearities import (
    LeakyRectify, rectify, sigmoid, softmax)
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

N_WORDS = 192
L_GEN = 32

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

class OneHotEncoding(SourcewiseTransformer):
    """Converts integer target variables to one hot encoding.

    It assumes that the targets are integer numbers from 0,... , N-1.
    Since it works on the fly the number of classes N needs to be
    specified.

    Batch input is assumed to be of shape (N,) or (N, 1).

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    num_classes : int
        The number of classes.

    """
    def __init__(self, data_stream, num_classes, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(OneHotEncoding, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.num_classes = num_classes

    def transform_source_example(self, source_example, source_name):
        if source_example >= self.num_classes:
            raise ValueError("source_example ({}) must be lower than "
                             "num_classes ({})".format(source_example,
                                                       self.num_classes))
        output = np.zeros((1, self.num_classes))
        output[0, source_example] = 1
        return output

    def transform_source_batch(self, source_batch, source_name):
        if np.max(source_batch) >= self.num_classes:
            raise ValueError("all entries in source_batch must be lower than "
                             "num_classes ({})".format(self.num_classes))
        shape = source_batch.shape
        source_batch = source_batch.flatten()
        output = np.zeros((source_batch.shape[0], self.num_classes),
                          dtype=source_batch.dtype)
        
        for i in range(self.num_classes):
            output[source_batch == i, i] = 1
        output = output.reshape((shape[0], shape[1], self.num_classes))
        return output.transpose(0, 2, 1)

def load_stream(batch_size=64, source=None):
    if source is None:
        raise ValueError('Source not set.')
    train_data = H5PYDataset(source, which_sets=('train',))
    train_scheme = ShuffledScheme(examples=train_data.num_examples, batch_size=batch_size)
    train_stream = OneHotEncoding(DataStream(train_data, iteration_scheme=train_scheme), N_WORDS)
    return train_stream, train_data.num_examples
            
# ##################### MODEL #####################

def build_generator(input_var=None, dim_h=512):
    layer = InputLayer(shape=(None, 100), input_var=input_var)

    # fully-connected layer
    #batch_norm = lambda x: x
    layer = batch_norm(DenseLayer(layer, dim_h * L_GEN, nonlinearity=None))
    layer = ReshapeLayer(layer, ([0], dim_h, L_GEN))
    for i in range(5):
        layer_ = NonlinearityLayer(layer, rectify)
        layer_ = batch_norm(Conv1DLayer(layer_, dim_h, 5, stride=1, pad=2))
        layer_ = batch_norm(Conv1DLayer(layer_, dim_h, 5, stride=1, pad=2, nonlinearity=None))
        layer = ElemwiseSumLayer([layer, layer_])
    layer = NonlinearityLayer(layer, rectify)
    layer = batch_norm(Conv1DLayer(layer, N_WORDS, 5, stride=1, pad=2, nonlinearity=None))
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def build_discriminator(input_var=None, dim_h=512):
    layer = InputLayer(shape=(None, N_WORDS, L_GEN), input_var=input_var)
    layer = Conv1DLayer(layer, dim_h, 5, stride=1, pad=2, nonlinearity=None)
    for i in range(5):
        layer_ = NonlinearityLayer(layer, lrelu)
        layer_ = batch_norm(Conv1DLayer(layer_, dim_h, 5, stride=1, pad=2))
        layer_ = batch_norm(Conv1DLayer(layer_, dim_h, 5, stride=1, pad=2, nonlinearity=None))
        layer = ElemwiseSumLayer([layer, layer_])
    layer = NonlinearityLayer(layer, lrelu)
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

def BGAN(discriminator, g_output_logit, n_samples, trng, batch_size=64):
    d = OrderedDict()
    d['g_output_logit'] = g_output_logit
    g_output_logit_ = g_output_logit.transpose(0, 2, 1)
    g_output_logit_ = g_output_logit_.reshape((-1, N_WORDS))
    d['g_output_logit_'] = g_output_logit_
    
    g_output = T.nnet.softmax(g_output_logit_)
    g_output = g_output.reshape((batch_size, L_GEN, N_WORDS))
    d['g_output'] = g_output
    
    p_t = T.tile(T.shape_padleft(g_output), (n_samples, 1, 1, 1))
    d['p_t'] = p_t
    p = p_t.reshape((-1, N_WORDS))
    d['p'] = p
    
    samples = trng.multinomial(pvals=p).astype(floatX)
    samples = theano.gradient.disconnected_grad(samples)
    samples = samples.reshape((n_samples, batch_size, L_GEN, N_WORDS))
    d['samples'] = samples
    
    D_r = lasagne.layers.get_output(discriminator)
    D_f = lasagne.layers.get_output(
        discriminator,
        samples.transpose(0, 1, 3, 2).reshape((-1, N_WORDS, L_GEN)))
    D_f_ = D_f.reshape((n_samples, -1))
    d.update(D_r=D_r, D_f=D_f, D_f_=D_f_)
    
    log_d1 = -T.nnet.softplus(-D_f_)
    log_d0 = -(D_f_ + T.nnet.softplus(-D_f_))
    log_w = D_f_
    d.update(log_d1=log_d1, log_d0=log_d0, log_w=log_w)

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    d['log_Z_est'] = log_Z_est

    g_output_logit = g_output_logit.transpose(0, 2, 1)
    log_g = (samples * (g_output_logit - log_sum_exp2(
        g_output_logit, axis=2))[None, :, :, :]).sum(axis=(2, 3))
    d['log_g'] = log_g
    
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
    d.update(log_w_tilde=log_w_tilde, w_tilde=w_tilde)
    
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-D_r)).mean() + (
        T.nnet.softplus(-D_f)).mean()
    
    return generator_loss, discriminator_loss, D_r, D_f, log_Z_est, d

# MAIN -------------------------------------------------------------------------

def summarize(results, samples, gt_samples, r_vocab, out_dir=None):
    results = dict((k, np.mean(v)) for k, v in results.items())    
    logger.info(results)
    
    gt_samples = np.argmax(gt_samples, axis=1)
    strs = []
    for gt_sample in gt_samples:
        s = ''.join([r_vocab[c] for c in gt_sample])
        strs.append(s)
    logger.info('GT:')
    logger.info(strs)
    
    samples = np.argmax(samples, axis=1)
    strs = []
    for sample in samples:
        s = ''.join([r_vocab[c] for c in sample])
        strs.append(s)
    logger.info('Samples:')
    logger.info(strs)

def main(source=None, vocab=None, num_epochs=None, gen_lr=None, beta_1_gen=None,
         dim_noise=None, initial_eta=None, batch_size=None, n_samples=None,
         beta_1_disc=None, disc_lr=None, num_iter_gen=None, n_steps=None,
         print_freq=None, image_dir=None, binary_dir=None, gt_image_dir=None,
         summary_updates=1000, debug=False):
    
    # Load the dataset
    stream, train_samples = load_stream(source=source, batch_size=batch_size)
    r_vocab = dict((v, k) for k, v in vocab.items())
    
    # VAR
    noise = T.matrix('noise')
    input_var = T.tensor3('inputs')
    
    # MODELS
    generator = build_generator(noise)
    discriminator = build_discriminator(input_var)
    trng = RandomStreams(random.randint(1, 1000000))

    # GRAPH / LOSS    
    g_output_logit = lasagne.layers.get_output(generator)
    
    generator_loss, discriminator_loss, D_r, D_f, log_Z_est, d = BGAN(
        discriminator, g_output_logit, n_samples, trng)

    if debug:
        batch = stream.get_epoch_iterator().next()[0]
        noise_ = lasagne.utils.floatX(np.random.rand(batch.shape[0],
                                                     dim_noise))
        print batch.shape
        for k, v in d.items():
            print 'Testing {}'.format(k)
            f = theano.function([noise, input_var], v, on_unused_input='warn')
            print k, f(noise_, batch.astype(floatX)).shape

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
        u = 0
        results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        
        for batch in stream.get_epoch_iterator():
            noise = lasagne.utils.floatX(np.random.rand(batch[0].shape[0],
                                                        dim_noise))
            outs = train_fn(batch[0].astype(floatX), noise)
            update_dict_of_lists(results, **outs)
            u += 1
            pbar.update(u)
            if u % summary_updates == 0:
                try:
                    samples = gen_fn(lasagne.utils.floatX(
                        np.random.rand(10, dim_noise)))
                    summarize(results, samples, batch[0][:10], r_vocab)
                except:
                    pass
        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        '''        
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta * 2 * (1 - progress)))
        '''

_defaults = dict(
    gen_lr=1e-4,
    beta_1_gen=0.5,
    beta_1_disc=0.5,
    disc_lr=1e-4,
    num_epochs=100,
    num_iter_gen=1,
    dim_noise=100,
    batch_size=64,
    n_samples=20,
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
    parser.add_argument('-V', '--vocab', type=str, default=None)
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

    with open('/data/lisa/data/1-billion-word/processed/one_billionr_voc_char.pkl') as f:
        vocab = pickle.load(f)
        
    main(source=args.source, vocab=vocab, **kwargs)
