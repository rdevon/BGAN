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
    batch_norm, Conv1DLayer, DenseLayer, ElemwiseSumLayer, Gate,
    GaussianNoiseLayer, InputLayer, LSTMLayer, NonlinearityLayer, ReshapeLayer)
from lasagne.nonlinearities import (
    LeakyRectify, rectify, sigmoid, softmax, tanh)
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
GRAD_CLIP = 100

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
        return output

def load_stream(batch_size=64, source=None):
    if source is None:
        raise ValueError('Source not set.')
    train_data = H5PYDataset(source, which_sets=('train',))
    train_scheme = ShuffledScheme(examples=train_data.num_examples, batch_size=batch_size)
    train_stream = OneHotEncoding(DataStream(train_data, iteration_scheme=train_scheme), N_WORDS)
    return train_stream, train_data.num_examples
            
# ##################### MODEL #####################

class BatchNormalizedLSTMLayer(LSTMLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 batch_axes=(0,),
                 **kwargs):

        # Initialize parent layer
        super(BatchNormalizedLSTMLayer, self).__init__(incoming, num_units,
                                                               ingate, forgetgate, cell, outgate,
                                                               nonlinearity, cell_init, hid_init,
                                                               backwards, learn_init, peepholes,
                                                               gradient_steps, grad_clipping,
                                                               unroll_scan, precompute_input, mask_input,
                                                               only_return_final, **kwargs)

        input_shape = self.input_shapes[0]

        # create BN layer with input shape (n_steps, batch_size, 4*num_units) and given axes
        self.bn_input = BatchNormLayer((input_shape[1], input_shape[0], 4*self.num_units), beta=None,
                                       gamma=init.Constant(0.1), axes=batch_axes)
        self.params.update(self.bn_input.params)  # make BN params your params

        # create batch normalization parameters for hidden units; the shape is (time_steps, num_units)
        self.epsilon = np.float32(1e-4)
        self.alpha = theano.shared(np.float32(0.1))
        shape = (input_shape[1], 4*num_units)
        self.gamma = self.add_param(init.Constant(0.1), shape, 'gamma', trainable=True, regularizable=True)
        self.running_mean = self.add_param(init.Constant(0), (input_shape[1], 4*num_units,), 'running_mean',
                                           trainable=False, regularizable=False)
        self.running_inv_std = self.add_param(init.Constant(1), (input_shape[1], 4*num_units,), 'running_inv_std',
                                              trainable=False, regularizable=False)
        self.running_mean_clone = theano.clone(self.running_mean, share_inputs=False)
        self.running_inv_std_clone = theano.clone(self.running_inv_std, share_inputs=False)
        self.running_mean_clone.default_update = self.running_mean_clone
        self.running_inv_std_clone.default_update = self.running_inv_std_clone

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        input = self.bn_input.get_output_for(T.dot(input, W_in_stacked)) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, gamma, time_step, cell_previous, hid_previous, *args):
            hidden = T.dot(hid_previous, W_hid_stacked)

            # batch normalization of hidden states
            if deterministic:
                mean = self.running_mean[time_step]
                inv_std = self.running_inv_std[time_step]
            else:
                mean = hidden.mean(0)
                inv_std = T.inv(T.sqrt(hidden.var(0) + self.epsilon))

                self.running_mean_clone.default_update = \
                    T.set_subtensor(self.running_mean_clone.default_update[time_step],
                        (1-self.alpha) * self.running_mean_clone.default_update[time_step] + self.alpha * mean)
                self.running_inv_std_clone.default_update = \
                    T.set_subtensor(self.running_inv_std_clone.default_update[time_step],
                        (1-self.alpha) * self.running_inv_std_clone.default_update[time_step] + self.alpha * inv_std)
                mean += 0 * self.running_mean_clone[time_step]
                inv_std += 0 * self.running_inv_std_clone[time_step]

            gamma = gamma.dimshuffle('x', 0)
            mean = mean.dimshuffle('x', 0)
            inv_std = inv_std.dimshuffle('x', 0)

            # normalize
            normalized = (hidden - mean) * (gamma * inv_std)

            # Calculate gates pre-activations and slice
            gates = input_n + normalized

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, gamma, time_step, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, gamma, time_step, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input, ]
            step_fun = step

        time_steps = np.asarray(np.arange(self.input_shapes[0][1]), dtype=np.int32)
        sequences.extend([self.gamma, time_steps])

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        non_seqs += [self.running_mean, self.running_inv_std]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

def build_generator(input_var=None, dim_h=128, n_steps=1):
    layer = InputLayer(shape=(None, None, 10), input_var=input_var)

    for i in range(n_steps):
        layer = BatchNormalizedLSTMLayer(
            layer, dim_h, grad_clipping=GRAD_CLIP,
            nonlinearity=tanh)
        layer = BatchNormalizedLSTMLayer(
            layer, dim_h, grad_clipping=GRAD_CLIP,
            nonlinearity=tanh, backwards=True)
        
    layer = LSTMLayer(
        layer, dim_h, grad_clipping=GRAD_CLIP, nonlinearity=tanh)
    layer = ReshapeLayer(layer, (-1, dim_h))
    layer = DenseLayer(layer, N_WORDS, nonlinearity=None)
    layer = ReshapeLayer(layer, (-1, L_GEN, N_WORDS))
    
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def build_discriminator(input_var=None, dim_h=128, n_steps=1):
    layer = InputLayer(shape=(None, None, N_WORDS), input_var=input_var)
    for i in range(n_steps):
        layer = LSTMLayer(
            layer, dim_h, grad_clipping=GRAD_CLIP,
            nonlinearity=tanh)
        layer = LSTMLayer(
            layer, dim_h, grad_clipping=GRAD_CLIP,
            nonlinearity=tanh)
    layer = ReshapeLayer(layer, (-1, dim_h))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    layer = ReshapeLayer(layer, (-1, L_GEN))
    
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
    g_output_logit_ = g_output_logit.reshape((-1, N_WORDS))
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
        samples.reshape((-1, L_GEN, N_WORDS)))
    D_f_ = D_f.reshape((n_samples, -1, L_GEN))
    d.update(D_r=D_r, D_f=D_f, D_f_=D_f_)
    
    log_d1 = -T.nnet.softplus(-D_f_)
    log_d0 = -(D_f_ + T.nnet.softplus(-D_f_))
    log_w = D_f_
    d.update(log_d1=log_d1, log_d0=log_d0, log_w=log_w)

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    d['log_Z_est'] = log_Z_est

    log_g = (samples * (g_output_logit - log_sum_exp2(
        g_output_logit, axis=2))[None, :, :, :]).sum(axis=3)
    d['log_g'] = log_g
    
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
    d.update(log_w_tilde=log_w_tilde, w_tilde=w_tilde)
    
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-D_r)).mean() + (
        T.nnet.softplus(-D_f)).mean() + D_f.mean()
    d.update(generator_loss=generator_loss,
             discriminator_loss=discriminator_loss)
    
    return generator_loss, discriminator_loss, D_r, D_f, log_Z_est, d

# MAIN -------------------------------------------------------------------------

def summarize(results, samples, gt_samples, r_vocab, out_dir=None):
    results = dict((k, np.mean(v)) for k, v in results.items())    
    logger.info(results)
    
    gt_samples = np.argmax(gt_samples, axis=2)
    strs = []
    for gt_sample in gt_samples:
        s = ''.join([r_vocab[c] for c in gt_sample])
        strs.append(s)
    logger.info('GT:')
    logger.info(strs)
    
    samples = np.argmax(samples, axis=2)
    strs = []
    for sample in samples:
        s = ''.join([r_vocab[c] for c in sample])
        strs.append(s)
    logger.info('Samples:')
    logger.info(strs)

def main(source=None, vocab=None, num_epochs=None, learning_rate=None, beta=None,
         dim_noise=None, batch_size=None, n_samples=None,
         n_steps=None,
         print_freq=None, image_dir=None, binary_dir=None, gt_image_dir=None,
         summary_updates=100, debug=False):
    
    # Load the dataset
    stream, train_samples = load_stream(source=source, batch_size=batch_size)
    r_vocab = dict((v, k) for k, v in vocab.items())
    
    # VAR
    noise = T.tensor3('noise')
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
                                                     L_GEN, dim_noise))
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
    
    l_kwargs = dict(learning_rate=learning_rate, beta1=beta)
    
    d_updates = lasagne.updates.adam(
        discriminator_loss, discriminator_params, **l_kwargs)
    g_updates = lasagne.updates.adam(
        generator_loss, generator_params, **l_kwargs)
    
    outputs = {
        'G cost': generator_loss,
        'D cost': discriminator_loss,
        'p(real)': T.nnet.sigmoid(D_r > .5).mean(),
        'p(fake)': T.nnet.sigmoid(D_f > .5).mean(),
    }
    gen_train_fn = theano.function([input_var, noise], outputs, updates=g_updates)
    disc_train_fn = theano.function([input_var, noise], [], updates=d_updates)
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
            noise = lasagne.utils.floatX(np.random.rand(
                batch[0].shape[0], L_GEN, dim_noise))
            disc_train_fn(batch[0].astype(floatX), noise)
            outs = gen_train_fn(batch[0].astype(floatX), noise)
            update_dict_of_lists(results, **outs)
            u += 1
            pbar.update(u)
            if u % summary_updates == 0:
                try:
                    samples = gen_fn(lasagne.utils.floatX(
                        np.random.rand(10, L_GEN, dim_noise)))
                    summarize(results, samples, batch[0][:10], r_vocab)
                except Exception as e:
                    print(e)
                    pass
        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))

_defaults = dict(
    learning_rate=1e-3,
    beta=0.5,
    num_epochs=100,
    dim_noise=10,
    batch_size=64,
    n_samples=20,
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
