'''Script for HGAN

'''

from collections import OrderedDict
from theano import tensor as T

import cortex
from cortex import set_experiment
from cortex.training.parsers import make_argument_parser
from cortex import _manager as manager
from cortex.utils.maths import norm_exp

def wasserman_cost_d(R, F):
    return F.mean() - R.mean()

def wasserman_cost_g(F):
    return -F.mean()

def alt_gen_cost(F, cells=None):
    d_name, = cells
    
    D_cell = manager.cells[d_name]

    log_py_h1 = -D_cell.neg_log_prob(1., P=F)
    log_py_h0 = -D_cell.neg_log_prob(0., P=F)
    log_p     = log_py_h1# - log_py_h0
    
    return -log_p.mean()

def reweighted_MLE(G_samples=None, cells=None):
    
    g_name, d_name = cells
    G_cell = manager.cells[g_name]    
    D_cell = manager.cells[d_name]

    d = D_cell(G_samples)['P']
    
    log_py_h1   = -D_cell.neg_log_prob(1., P=d)
    log_py_h0   = -D_cell.neg_log_prob(0., P=d)

    log_p       = log_py_h1 - log_py_h0
    w = T.exp(log_p)
    w_tilde = norm_exp(log_p)
    cost = -(w_tilde[:, None] * G_samples).sum(0).mean()
    return OrderedDict(cost=cost, constants=[w_tilde])

def main(batch_size=None, dim_z=None, GAN_type=None, freq=1, test=False):
    cortex.set_path('HGAN')
    '''
    cortex.prepare_data('euclidean', name='data', method_args=dict(N=2),
                        n_samples=10000, method='modes', mode='train')
    cortex.prepare_data('euclidean', name='data', method_args=dict(N=2),
                        n_samples=10000, method='modes', mode='valid')
    '''
    source = '$data/basic/mnist.pkl.gz'
    cortex.prepare_data('MNIST', mode='train', name='data', source=source)
    cortex.prepare_data('MNIST', mode='valid', name='data', source=source)

    cortex.prepare_cell('gaussian', name='noise', dim=dim_z)
    
    d_args = dict(
        dim_hs=[500, 200],
        h_act='softplus',
        dropout=0.2,
        name='discriminator'
    )
    if GAN_type is None:
        cortex.prepare_cell('DistributionMLP', distribution_type='binomial',
                            dim=1, **d_args)
    elif GAN_type.lower() == 'wasserstein':
        cortex.prepare_cell('MLP', dim_out=1, **d_args)
        
    g_args = dict(
        h_act='softplus',
        out_act='sigmoid',
        dim_hs=[500, 500],
        batch_normalization=True,
        name='generator'
    )
    cortex.prepare_cell('MLP', **g_args)
    
    cortex.add_step('discriminator', 'data.input', name='real')
    cortex.prepare_samples('noise', batch_size)
    cortex.add_step('generator', 'noise.samples')
    cortex.add_step('discriminator', 'generator.output', name='fake')
    
    if GAN_type is None:
        cortex.add_step('discriminator._cost', P='fake.P', X=0.,
                        name='fake_cost')
        cortex.add_step('discriminator._cost', P='real.P', X=1.,
                        name='real_cost')

        cortex.build()
        
        cortex.add_cost(
            lambda x, y: x + y, 'fake_cost.output', 'real_cost.output',
            name='discriminator_cost')
        '''
        cortex.add_cost('discriminator.negative_log_likelihood',
                        X=1., P='fake.P', name='generator_cost')
        
        cortex.add_cost(alt_gen_cost, F='fake.P',
                        cells=['discriminator'],
                        name='generator_cost')
        '''
        cortex.add_cost(
            reweighted_MLE, G_samples='generator.output',
            cells=['generator', 'discriminator'], name='generator_cost')
        
        cortex.add_stat('basic_stats', 'fake.P', name='fake_rate')
        cortex.add_stat('basic_stats', 'real.P', name='real_rate')
        
    elif GAN_type.lower() == 'wasserstein':
        cortex.build()
        cortex.add_cost(wasserman_cost_d, 'real.output', 'fake.output',
                        name='discriminator_cost')
        cortex.add_cost(wasserman_cost_g, 'fake.output',
                        name='generator_cost')
    cortex.add_stat('basic_stats', 'noise.samples', name='noise')
    
    train_session = cortex.create_session()
    cortex.build_session()
    
    trainer = cortex.setup_trainer(
        train_session,
        optimizer='sgd',
        epochs=1000,
        learning_rate=0.01,
        batch_size=batch_size,
    )
    
    if GAN_type is None:
        optimizer_args = {}
    elif GAN_type.lower() == 'wasserstein':
        optimizer_args = {
            'clips': {'discriminator.weights[0]_grad': 0.01,
                      'discriminator.weights[1]_grad': 0.01}
        }
        
    trainer.set_optimizer(
        ['discriminator', 'discriminator_cost'],
        optimizer='sgd', optimizer_args=optimizer_args)
    trainer.set_optimizer(
        ['generator', 'generator_cost'],
        freq=freq, optimizer='sgd')
    
    valid_session = cortex.create_session(noise=False)
    cortex.build_session()
    
    evaluator = cortex.setup_evaluator(valid_session, valid_stat='total_cost')
    
    monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])
    
    visualizer = cortex.setup_visualizer(train_session, batch_size=batch_size)
    visualizer.add('data.viz',
                   X='generator.output',
                   name='generated')
    visualizer.add('data.viz',
                   X='data.input',
                   name='real')
    
    cortex.train(monitor_grads=False)
    

if __name__ == '__main__':
    parser = make_argument_parser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-d', '--dim_z', type=int, default=500)

    args = parser.parse_args()
    kwargs = set_experiment(args)
    main(**kwargs)