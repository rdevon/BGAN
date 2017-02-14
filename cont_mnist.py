'''Script for HGAN

'''

from collections import OrderedDict
from theano import tensor as T

import cortex
from cortex import set_experiment
from cortex.training.parsers import make_argument_parser
from cortex import _manager as manager
from cortex.utils import floatX
from cortex.utils.maths import norm_exp, log_sum_exp

def wasserman_cost_d(R, F):
    return dict(cost=F.mean() - R.mean(), r=R.mean(), f=F.mean())

def wasserman_cost_g(F):
    return dict(cost=-F.mean(), f=F.mean())

def alt_gen_cost(F, cells=None):
    d_name, = cells
    D_cell = manager.cells[d_name]

    log_py_h1 = -D_cell.neg_log_prob(1., P=F)
    log_py_h0 = -D_cell.neg_log_prob(0., P=F)
    log_p     = log_py_h1 - log_py_h0
    
    return -log_p.mean()

def gen_cost(F, cells=None):
    d_name, = cells
    D_cell = manager.cells[d_name]
    log_py_h1 = -D_cell.neg_log_prob(1., P=F)
    
    return -log_py_h1.mean()

def reweighted_MLE(G_samples=None, cells=None):
    d_name, = cells
    D_cell = manager.cells[d_name]
    d = D_cell(G_samples)['P']
    
    log_d1 = T.log(d) #-D_cell.neg_log_prob(1., P=d)
    log_d0 = T.log(1. - d)#-D_cell.neg_log_prob(0., P=d)
    log_w = log_d1 - log_d0
    
    # Find normalized weights.
    log_N = T.log(log_w.shape[0]).astype(floatX)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_w_sum = log_sum_exp(log_w, axis=0)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)

    ess = 1. / (w_tilde ** 2).sum(0)
    log_ess = (-T.log((w_tilde ** 2).sum(0)))    
    
    #cost = -(w_tilde * log_w).sum(0).mean()
    #constants = [w_tilde]
    
    scale = (w_tilde * (log_w - T.maximum(log_Z_est, -1) + 1)).copy()
    #scale = -log_w.copy()
    cost = (scale * log_w).sum(0).mean()
    constants = [scale, w_tilde, log_Z_est]
    
    #cost = (w_tilde * (log_w - T.maximum(log_w_norm, 0.9))).sum()
    #constants = [log_w_norm]
    #cost = -log_w_tilde.mean()
    #scale = w_tilde * (log_N + log_w_tilde + 1.)
    #constants = [scale]
    #cost = (scale * log_w_tilde).mean()
    
    return OrderedDict(cost=cost,
                       w_tilde=w_tilde.mean(),
                       log_N=log_N,
                       log_Z_est=log_Z_est,
                       log_w_tilde=log_w_tilde.mean(),
                       log_w_tilde2=T.log(w_tilde).mean(),
                       log_w_tilde_max=log_w_tilde.max(),
                       diff_log=(T.log(w_tilde) + log_N).mean(),
                       diff_log2=(log_w_tilde + log_N).mean(), 
                       w_tilde_std=w_tilde.std(),
                       d=d.mean(),
                       log_d1=log_d1, log_d0=log_d0,
                       w=T.exp(log_w).mean(),
                       ess=ess, log_ess=log_ess,
                       constants=constants)

def build_normal_GAN():
    cortex.add_step('discriminator._cost', P='fake.P', X=0.,
                    name='fake_cost')
    cortex.add_step('discriminator._cost', P='real.P', X=1.,
                    name='real_cost')

    cortex.build()
    
    cortex.add_cost(
        lambda x, y: x + y, 'fake_cost.output', 'real_cost.output',
        name='discriminator_cost')
    
    m = 0
    if m == 0:
        cortex.add_cost(gen_cost, F='fake.P',
                        cells=['discriminator'],
                        name='generator_cost')
    elif m == 1:    
        cortex.add_cost(alt_gen_cost, F='fake.P',
                        cells=['discriminator'],
                        name='generator_cost')
    elif m == 2:
        cortex.add_cost(
            reweighted_MLE, G_samples='generator.output',
            cells=['discriminator'], name='generator_cost')
    else:
        raise

    cortex.add_stat('basic_stats', 'fake.P', name='fake_rate')
    cortex.add_stat('basic_stats', 'real.P', name='real_rate')

def build_wasserstein_GAN():
    cortex.build()
    cortex.add_cost(wasserman_cost_d, 'real.output', 'fake.output',
                    name='discriminator_cost')
    cortex.add_cost(wasserman_cost_g, 'fake.output',
                    name='generator_cost')
    cortex.add_stat('basic_stats', 'data.input')

def main(batch_size=None, dim_z=None, GAN_type=None, freq=1,
         learning_rate=0.01, optimizer='sgd', test=False):
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
        name='discriminator',
        dropout=0.1
    )
    if GAN_type is None:
        cortex.prepare_cell('DistributionMLP', distribution_type='binomial',
                            dim=1, **d_args)
    elif GAN_type.lower() == 'wasserstein':
        cortex.prepare_cell('MLP', dim_out=1, out_act='identity', **d_args)        
    else:
        raise
        
    g_args = dict(
        h_act='softplus',
        out_act='sigmoid',
        dim_hs=[500, 500, 500],
        batch_normalization=True,
        name='generator'
    )
    cortex.prepare_cell('MLP', **g_args)
    
    cortex.add_step('discriminator', 'data.input', name='real')
    
    cortex.prepare_samples('noise', batch_size)
    cortex.add_step('generator', 'noise.samples')
    cortex.add_step('discriminator', 'generator.output', name='fake')
    
    if GAN_type is None:
        build_normal_GAN()
        optimizer_args = {}
        '''
        clip = 0.01
        optimizer_args = {            
            'clips': {'discriminator.weights[0]': clip,
                      'discriminator.weights[1]': clip,
                      'discriminator.weights[2]': clip}
        }
        '''
        
    elif GAN_type.lower() == 'wasserstein':
        build_wasserstein_GAN()
        clip = 0.01
        optimizer_args = {            
            'clips': {'discriminator.weights[0]': clip,
                      'discriminator.weights[1]': clip,
                      'discriminator.weights[2]': clip}
        }
    else:
        raise
    
    train_session = cortex.create_session()
    cortex.build_session()
    
    trainer = cortex.setup_trainer(
        train_session,
        optimizer=optimizer,
        epochs=10000,
        learning_rate=learning_rate,
        learning_rate_decay=.99,
        batch_size=batch_size,
    )
        
    trainer.set_optimizer(
        ['discriminator', 'discriminator_cost'], optimizer_args=optimizer_args,)
        #grad_clip=dict(clip_type='norm', clip_norm=10))
    trainer.set_optimizer(
        ['generator', 'generator_cost'], freq=freq)
    
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
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-d', '--dim_z', type=int, default=500)

    args = parser.parse_args()
    kwargs = set_experiment(args)
    main(**kwargs)
