'''Losses

'''

import logging

import lasagne
import theano
from theano import tensor as T

from math import log_sum_exp
from utils import trng


logger = logging.getLogger('BGAN.loss')
_default_clip = 0.01
_default_log_Z_gamma = 0.995
floatX = theano.config.floatX


def get_losses(real_out, fake_out, loss=None, loss_options=None,
               optimizer_args=None):
    loss_options = loss_options or {}
    logger.info('Using loss `{}` with args {}'.format(loss, loss_options))
    
    loss_dict = _losses
    
    if callable(loss):
        logger.info('Using custom loss')
        g_results, d_results = loss(fake_out, real_out, **loss_options)
    else:
        loss_fn = loss_dict.get(loss, None)
        if loss_fn is None:
            raise NotImplementedError('Unknown loss function `{}`. Available: '
                                      '{}'.format(loss, loss_dict.keys()))
        
    g_results, d_results = loss_fn(fake_out, real_out, **loss_options)
    
    if optimizer_args is not None:
        if loss == 'wgan' and 'clip' not in optimizer_args.keys():
            logger.warning('Adding default clipping to WGAN, {}'.format(
                _default_clip))
            optimizer_args['clip'] = _default_clip
        if ('log_Z' in loss_options.keys()
            and 'log_Z_gamma' not in optimizer_args.keys()):
            logger.warning('Using default log Z gamma of {}'.format(
                _default_log_Z_gamma))
            optimizer_args['log_Z_gamma'] = _default_log_Z_gamma

    return g_results, d_results


def get_losses_discrete(discriminator, g_output_logit, loss=None,
                        optimizer_args=None, **loss_args):
    loss_dict = _disc_losses
    loss_args.pop('loss_options')
    
    if callable(loss):
        logger.info('Using custom loss')
        g_results, d_results, log_Z_est = loss(discriminator, g_output_logit,
                                               **loss_args)
    else:
        loss_fn = loss_dict.get(loss, None)
        if loss_fn is None:
            raise NotImplementedError('Unknown loss function `{}`. Available: '
                                      '{}'.format(loss, loss_dict.keys()))
        
    g_results, d_results, log_Z_est = loss_fn(discriminator, g_output_logit,
                                              **loss_args)
    
    if optimizer_args is not None:
        if 'log_Z_gamma' not in optimizer_args.keys():
            logger.warning('Using default log Z gamma of {}'.format(
                _default_log_Z_gamma))
            optimizer_args['log_Z_gamma'] = _default_log_Z_gamma
    
    return g_results, d_results, log_Z_est


def BGAN(fake_out, real_out, log_Z=None, use_log_Z=False,
         use_cross_entropy=False):
    '''Boundary-seeking GAN loss.
    
    '''
    if use_log_Z:
        logger.info('Using log Z in BGAN')
        generator_loss = ((fake_out - log_Z) ** 2).mean()
    elif use_cross_entropy:
        logger.info('Using cross entropy loss in BGAN')
        generator_loss = (T.nnet.softplus(-fake_out) + 0.5 * fake_out).mean()
    else:
        generator_loss = (fake_out ** 2).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    
    g_results = {
        'g loss': generator_loss,
        'p(fake==0)': 1. - T.nnet.sigmoid(fake_out).mean(),
        'E[logit(fake)]': fake_out.mean()
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'p(real==1)': T.nnet.sigmoid(real_out).mean(),
        'E[logit(real)]': real_out.mean(),
    }
    
    return g_results, d_results


def LSGAN(fake_out, real_out, target=1.0):
    '''Least squares GAN loss
    
    '''
    logger.info('Generator target is {}'.format(target))
    generator_loss = ((fake_out - target) ** 2).mean()
    discriminator_loss = 0.5 * ((real_out - 1.) ** 2).mean() + 0.5 * (
        fake_out ** 2).mean()
    
    g_results = {
        'g loss': generator_loss,
        'E[fake]': fake_out.mean()
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'E[real]': real_out.mean()
    }
    
    return g_results, d_results


def WGAN(fake_out, real_out):
    '''Wasserstein GAN loss
    
    '''
    generator_loss = -fake_out.mean()
    discriminator_loss = -real_out.mean() + fake_out.mean()
    
    g_results = {
        'g loss': generator_loss,
        'E[fake]': fake_out.mean()
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'E[real]': real_out.mean()
    }
    
    return g_results, d_results


def GAN(fake_out, real_out):
    '''Loss for normal generative adversarial networks.
    
    '''
    generator_loss = T.nnet.softplus(-fake_out).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        (T.nnet.softplus(-fake_out) + fake_out)).mean()
    
    g_results = {
        'g loss': generator_loss,
        'p(fake==0)': 1. - T.nnet.sigmoid(fake_out).mean(),
        'E[logit(fake)]': fake_out.mean()
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'p(real==1)': T.nnet.sigmoid(real_out).mean(),
        'E[logit(real)]': real_out.mean(),
    }
    
    return g_results, d_results


def binary_BGAN(discriminator, g_output_logit, n_samples=None, log_Z=None,
                batch_size=None, dim_c=None, dim_x=None, dim_y=None):
    '''Discrete BGAN for discrete binary variables.
    
    '''
    # Sample from a uniform distribution and generate samples over input.
    R = trng.uniform(size=(n_samples, batch_size, dim_c, dim_x, dim_y),
                     dtype=floatX)
    g_output = T.nnet.sigmoid(g_output_logit)
    samples = (R <= T.shape_padleft(g_output)).astype(floatX)

    # Feed samples through the discriminator.
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, samples.reshape((-1, dim_c, dim_x, dim_y)))
    log_w = fake_out.reshape((n_samples, batch_size))
    
    # Get the log probabilities of the samples.
    log_g = -((1. - samples) * T.shape_padleft(g_output_logit)
             + T.shape_padleft(T.nnet.softplus(-g_output_logit))).sum(
        axis=(2, 3, 4))
    
    # Get the normalized weights.
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)

    # Losses.
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (
        T.nnet.softplus(-fake_out)).mean() + fake_out.mean()
    
    g_results = {
        'g loss': generator_loss,
        'p(fake==0)': 1. - T.nnet.sigmoid(fake_out).mean(),
        'E[logit(fake)]': fake_out.mean(),
        'log w': log_w.mean(),
        'log w var': log_w.std() ** 2,
        'norm w': w_tilde.mean(),
        'norm w var': w_tilde.std() ** 2,
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean(),
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'p(real==1)': T.nnet.sigmoid(real_out).mean(),
        'E[logit(real)]': real_out.mean(),
    }
    
    return g_results, d_results, log_Z_est


def multinomial_BGAN(discriminator, g_output_logit, n_samples=None, log_Z=None,
                     batch_size=None, dim_c=None, dim_x=None, dim_y=None):
    
    g_output_logit_ = g_output_logit.transpose(0, 2, 3, 1)
    g_output_logit_ = g_output_logit_.reshape((-1, dim_c))
    
    g_output = T.nnet.softmax(g_output_logit_)
    g_output = g_output.reshape((batch_size, dim_x, dim_y, dim_c))
    
    p_t = T.tile(T.shape_padleft(g_output), (n_samples, 1, 1, 1, 1))
    p = p_t.reshape((-1, dim_c))
    
    samples = trng.multinomial(pvals=p).astype(floatX)
    samples = theano.gradient.disconnected_grad(samples)
    samples = samples.reshape((n_samples, batch_size, dim_x, dim_y, dim_c))
    samples = samples.transpose(0, 1, 4, 2, 3)
    
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, samples.reshape((-1, dim_c, dim_x, dim_y)))
    log_w = fake_out.reshape((n_samples, -1))

    log_g = ((samples * (g_output_logit - log_sum_exp(
        g_output_logit, axis=1, keepdims=True))[None, :, :, :, :])
        .sum(axis=(2, 3, 4)))
    
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
    
    generator_loss = -(w_tilde_ * log_g).sum(0).mean()
    discriminator_loss = (T.nnet.softplus(-real_out)).mean() + (
        T.nnet.softplus(-fake_out)).mean() + fake_out.mean()

    g_results = {
        'g loss': generator_loss,
        'p(fake==0)': 1. - T.nnet.sigmoid(fake_out).mean(),
        'E[logit(fake)]': fake_out.mean(),
        'log w': log_w.mean(),
        'log w var': log_w.std() ** 2,
        'norm w': w_tilde.mean(),
        'norm w var': w_tilde.std() ** 2,
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean(),
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'p(real==1)': T.nnet.sigmoid(real_out).mean(),
        'E[logit(real)]': real_out.mean(),
    }
    
    return g_results, d_results, log_Z_est


def IDB(discriminator, g_output_logit, n_samples=None, log_Z=None,
        batch_size=None, dim_c=None, dim_x=None, dim_y=None):
    R = trng.uniform(size=(n_samples, batch_size, dim_c, dim_x, dim_y),
                     dtype=floatX)
    g_output = T.nnet.sigmoid(g_output_logit)
    samples = (R <= T.shape_padleft(g_output)).astype(floatX)
    
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, samples.reshape((-1, dim_c, dim_x, dim_y)))
    log_w = fake_out.reshape((n_samples, batch_size))
    
    log_g = -((1. - samples) * T.shape_padleft(g_output_logit)
             + T.shape_padleft(T.nnet.softplus(-g_output_logit))).sum(
        axis=(2, 3, 4))
    
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
    r = theano.gradient.disconnected_grad((log_w - log_Z[None, :] - 1))
    
    generator_loss = -(r * log_g).mean()
    discriminator_loss = (T.nnet.softplus(-D_r)).mean() + (
        T.nnet.softplus(-D_f)).mean() + D_f.mean()
    
    g_results = {
        'g loss': generator_loss,
        'p(fake==0)': 1. - T.nnet.sigmoid(fake_out).mean(),
        'E[logit(fake)]': fake_out.mean(),
        'log w': log_w.mean(),
        'log w var': log_w.std() ** 2,
        'norm w': w_tilde.mean(),
        'norm w var': w_tilde.std() ** 2,
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean()
    }
    
    d_results = {
        'd loss': discriminator_loss,
        'p(real==1)': T.nnet.sigmoid(real_out).mean(),
        'E[logit(real)]': real_out.mean(),
    }
    
    return g_results, d_results, log_Z_est


_disc_losses = dict(
    binary_bgan=binary_BGAN,
    multinomial_bgan=multinomial_BGAN
)

_losses = dict(
    bgan=BGAN,
    lsgan=LSGAN,
    gan=GAN,
    wgan=WGAN
)