'''GAN with RWS.

'''

from collections import OrderedDict
import cortex
from cortex import set_experiment
from cortex.training.parsers import make_argument_parser
from cortex.utils.maths import norm_exp
from cortex import _manager as manager

from theano import tensor as T


def reweighted_MLE(G=None, G_samples=None, cells=None):
    
    if G is None:
        raise TypeError('Generator distribution (G) must be provided.')
    
    g_name, d_name = cells
    
    if isinstance(G, str):
        G_cell = manager.cells[G]
        G = G_cell.get_prob(*G_cell.get_params())
    else:
        G_cell = manager.cells[g_name]
        
    D_cell = manager.cells[d_name]

    d = D_cell.feed_batch(G_samples, 64*20)['P']
    
    log_py_h1   = -D_cell.neg_log_prob(1., P=d)
    log_py_h0   = -D_cell.neg_log_prob(0., P=d)
    log_gx      = -G_cell.neg_log_prob(G_samples, P=G[None, :, :])

    log_p       = log_py_h1 - log_py_h0
    w = T.exp(log_p)
    w_tilde = norm_exp(log_p)
    cost = -(w_tilde * log_gx).sum(0).mean()
    return OrderedDict(cost=cost, constants=[w_tilde])

def main(name='airrws', data='MNIST', batch_size=64, dim_in=200,
         n_posterior_samples=20, test=False):
    
    if data == 'MNIST':
        source = '$data/basic/mnist_binarized_salakhutdinov.pkl.gz'
        distribution_type = 'binomial'
        dim = 28 * 28
        greyscale = True
        filters = 1
    elif data == 'CIFAR':
        greyscale = True
        source='$data/basic/cifar-10-batches-py/'
        distribution_type = 'gaussian_unit_variance'
        dim = 32 * 32
        filters = 3
        if not greyscale: dim *= 3
        
    d_dropout = 0.
    d2_dropout = 0.2

    cortex.set_path(name)

    # DATA ---------------------------------------------------------------------
    cortex.prepare_data(data, mode='train', source=source, name='data',
                        greyscale=greyscale)
    cortex.prepare_data(data, mode='valid', source=source, name='data',
                        greyscale=greyscale)
    cortex.prepare_data(data, mode='test', source=source, name='data',
                        greyscale=greyscale)
    
    # CELLS --------------------------------------------------------------------

    # Generative model
    cortex.prepare_cell('gaussian', name='noise', dim=dim_in)
    
    generator = dict(
        cell_type='RCNN2D',
        input_shape=(256, 5, 5),
        filter_shapes=((5, 5), (4, 4), (4, 4)),
        strides=((1, 1), (2, 2), (2, 2)),
        pads=((1, 1), (1, 1), (1, 1)),
        n_filters=[128, 64, filters], h_act='softplus', dim_in=dim_in,
        batch_normalization=True, out_act='sigmoid')
    
    discriminator = dict(
        cell_type='CNN2D',
        input_shape=cortex._manager.datasets['data']['image_shape'],
        filter_shapes=((4, 4), (4, 4), (5, 5)),
        strides=((2, 2), (2, 2), (1, 1)),
        pads=((1, 1), (1, 1), (1, 1)),
        n_filters=[64, 128, 256], dim_out=1, h_act='softplus',
        batch_normalization=False)
    
    cortex.prepare_cell('DistributionMLP', name='discriminator',
                        mlp=discriminator,
                        dim=1,
                        distribution_type='binomial')
    
    cortex.prepare_cell('DistributionMLP', name='generator',
                        mlp=generator,
                        dim=dim,
                        distribution_type=distribution_type)
    
    # GRAPH --------------------------------------------------------------------

    cortex.add_step('discriminator', 'data.input', name='real')

    cortex.prepare_samples('noise', batch_size)
    cortex.add_step('generator', 'noise.samples', constants=['noise.samples'])
    cortex.prepare_samples('generator.P', n_posterior_samples)
    cortex.add_step('discriminator.feed_batch', 'generator.samples',
                    batch_size*n_posterior_samples,
                    name='fake', constants=['generator.samples'])
    
    cortex.add_step('discriminator._cost', P='fake.P', X=0., name='fake_cost')
    cortex.add_step('discriminator._cost', P='real.P', X=1., name='real_cost')
    
    cortex.build()

    #cortex.add_cost('l2_decay', 0.002, 'discriminator.mlp.weights')
    #cortex.add_cost('l2_decay', 0.002, 'generator.mlp.weights')
    
    cortex.add_cost(lambda x, y: x + y, 'fake_cost.output', 'real_cost.output',
                name='discriminator_cost')
    cortex.add_cost(reweighted_MLE, G='generator.P',
                    G_samples='generator.samples',
                    cells=['generator', 'discriminator'],
                    name='generator_cost')
    
    train_session = cortex.create_session(batch_size=batch_size)
    cortex.build_session(test=test)

    trainer = cortex.setup_trainer(
        train_session,
        optimizer='rmsprop',
        epochs=3000,
        learning_rate=0.0001,
        batch_size=batch_size,
        excludes=['noise.mu', 'noise.log_sigma'])

    model_costs = [
        (['discriminator.mlp', 'discriminator.distribution'], 'discriminator_cost'),
        (['generator.mlp', 'generator.distribution'], 'generator_cost')]

    trainer.set_optimizer(*model_costs)

    valid_session = cortex.create_session(noise=False, batch_size=batch_size)
    cortex.build_session()

    evaluator = cortex.setup_evaluator(
        valid_session,
        valid_stat='generator_cost',
        batch_size=batch_size)

    monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

    visualizer = cortex.setup_visualizer(valid_session, batch_size=100)
    visualizer.add('data.viz', X='generator.P_center', name='AIRGAN_gen')
    visualizer.add('data.viz', X='data.input', name='data_sample')

    cortex.train(eval_every=10, archive_every=100)

if __name__ == '__main__':

    parser = make_argument_parser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-D', '--dim_in', type=int, default=200)
    parser.add_argument('-p', '--n_posterior_samples', type=int, default=20)
    parser.add_argument('-d', '--data', type=str, default='MNIST')

    args = parser.parse_args()

    kwargs = set_experiment(args)
    main(**kwargs)