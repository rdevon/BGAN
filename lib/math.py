'''Math.

'''

import theano
from theano import tensor as T


def log_sum_exp(x, axis=None, keepdims=False):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis, keepdims=keepdims)
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