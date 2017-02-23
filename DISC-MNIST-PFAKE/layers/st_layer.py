import lasagne
import lasagne.nonlinearities
import theano
import theano.tensor as T
from ops.gradient_switch_op import gradient_switch_op
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class STLayer(lasagne.layers.Layer):
    def __init__(self, incoming, K, **kwargs):
        super(STLayer, self).__init__(incoming, **kwargs)
        self.K = K
        self.trng = RandomStreams(12345)


    def get_output_for(self, _input, **kwargs):
        input_reshaped_1 = T.reshape(_input, (-1, 1))
        input_reshaped_2 = 1 - input_reshaped_1
        input_reshaped = T.concatenate([input_reshaped_1, input_reshaped_2], axis=1)
        concept = self.trng.multinomial(pvals=input_reshaped, dtype='float32')
        concept_disc = theano.gradient.disconnected_grad(concept)
        concept_disc = T.reshape(concept_disc[:, 0], (-1, 1))
        concept_disc = gradient_switch_op(concept_disc, input_reshaped_1)
        output = T.reshape(concept_disc, (-1, 1, self.K, self.K))
        return output
