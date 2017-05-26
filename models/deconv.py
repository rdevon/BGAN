'''Alternate deconvolution.

'''

import lasagne
from theano import tensor as T


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
                W, (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        if b is None:
            self.b = self.add_param(
                lasagne.init.Constant(0), (num_filters,), name='b')
        else:
            self.b = self.add_param(
                b, (num_filters,), name='b')
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