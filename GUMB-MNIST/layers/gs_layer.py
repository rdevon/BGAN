import theano
import lasagne
import theano.tensor as T
from ops.gumbel_softmax import gumbel_softmax
from ops.gradient_switch_op import gradient_switch_op
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class GumbelSoftmaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, temperature, K, hard="false",**kwargs):
        super(GumbelSoftmaxLayer, self).__init__(incoming, **kwargs)
        self.trng = RandomStreams(12345)
        self.hard = hard
        self.K = K
        self.temperature = temperature

    def get_output_for(self, input_, **kwargs):
        input_reshaped = T.reshape(input_, (-1, self.K * self.K))
        concept_disc = gumbel_softmax(input_reshaped, self.trng, temperature=self.temperature, hard=self.hard)
        output = T.reshape(concept_disc, (-1, 1, self.K, self.K))
        return output

