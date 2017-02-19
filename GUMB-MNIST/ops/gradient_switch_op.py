import theano
from theano import tensor
from theano import tensor as T
from theano.gof import Apply
from theano.tensor import as_tensor_variable
from theano.tensor import basic as tensor


class Gradient_Switch_Op(theano.Op):

    __props__ = ("odtype",)

    def __init__(self, odtype):
        self.odtype = odtype

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def make_node(self, h1, h2):
        h1 = as_tensor_variable(h1)
        h2 = as_tensor_variable(h2)
        otype = tensor.tensor(
            broadcastable=(False, False),
            dtype='float32')
        return Apply(self, [h1, h2], [otype])

    def perform(self, node, inputs, output_storage):
        h1, h2 = inputs
        output_storage[0][0] = h1

    def grad(self, inp, grads):
        h1, h2 = inp
        gz, = grads
        return gz, gz

gradient_switch_op = Gradient_Switch_Op('float32')
