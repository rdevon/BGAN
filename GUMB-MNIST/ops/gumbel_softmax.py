import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def sample_gumbel(rng, size, eps=1e-20):
    U = rng.uniform(size)
    return -T.log(-T.log(U + eps) + eps)

def gumbel_softmax_sample(incoming, rng, temperature):
    y = incoming + sample_gumbel(rng, T.shape(incoming))
    return T.nnet.softmax(y / temperature)

def gumbel_softmax(incoming, rng, temperature, hard=False, dtype=None):
    y = gumbel_softmax_sample(incoming, rng, temperature)

    if dtype is None:
        dtype = y.dtype
    if hard:
        y_hard = T.cast(T.eq(y, T.max(y, axis=1, keepdims=True)), dtype)
        y = theano.gradient.disconnected_grad(y_hard - y) + y

    return y
