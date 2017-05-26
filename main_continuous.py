'''Main function to train continuous BGAN.

'''

import logging

import lasagne
import numpy as np
import theano
import theano.tensor as T

from lib.data import load_stream
from lib.log_util import set_stream_logger
from lib.loss import get_losses
from lib.math import est_log_Z
from lib.train import setup, train
from lib.utils import config, make_argument_parser, print_section, setup_out_dir
from lib.viz import setup as setup_viz
from models import build


logger = logging.getLogger('BGAN')


def main(data_args=None, optimizer_args=None, model_args=None, loss_args=None,
         train_args=None):
    '''Main function for continuous BGAN.
    
    '''
      
    print_section('LOADING DATA') ##############################################
    train_stream, training_samples, shape, viz_options = load_stream(
        **data_args)
    train_args['training_samples'] = training_samples
    setup_viz(**viz_options)
    model_args.update(**shape)
    
    print_section('MODEL') #####################################################
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    
    if loss_args['loss'] == 'bgan':
        log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')
        loss_args['loss_options']['log_Z'] = log_Z
    else:
        log_Z = None

    logger.info('Building model and compiling GAN functions...')
    logger.info('Model args: {}'.format(model_args))
    generator, discriminator = build(noise_var, input_var, **model_args)

    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(
        discriminator, lasagne.layers.get_output(generator))
    
    g_results, d_results = get_losses(
        real_out, fake_out, optimizer_args=optimizer_args, **loss_args)
    
    if log_Z is not None:
        log_Z_est = est_log_Z(fake_out)
        g_results.update(**{
            'log Z': log_Z,
            'log Z (est)': log_Z_est.mean()
        })
        
    print_section('OPTIMIZER') #################################################
    train_d, train_g, gen =  setup(input_var, noise_var, log_Z, generator,
                                   discriminator, g_results, d_results,
                                   **optimizer_args)
        
    print_section('TRAIN') #####################################################
    try:
        train(train_d, train_g, gen, train_stream, **train_args)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
        print_section('DONE') ##################################################
        exit(0)


_default_args = dict(
    data_args=dict(
        batch_size=64,
        use_tanh=True
    ),
    optimizer_args=dict(
        optimizer='adam',
        optimizer_options=dict(
            learning_rate=1e-3,
            beta1=0.5
        )
    ),
    model_args=dict(
        arch='dcgan_64',
        dim_z=128,
        dim_h=128,
        leak=0.02 
    ),
    loss_args=dict(
        loss='bgan',
        loss_options=dict(use_log_Z=True)
    ),
    train_args=dict(
        epochs=50,
        num_iter_gen=1,
        num_iter_disc=1,
        summary_updates=None,
        archive_every=10
    )
)


if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    
    kwargs = {}
    for k, v in _default_args.items():
        kwargs[k] = {}
        kwargs[k].update(**v)
    
    kwargs['data_args']['source'] = args.source
    if args.architecture is not None:
        kwargs['model_args']['arch'] = args.architecture
        
    out_paths = setup_out_dir(args.out_path, args.name)
    kwargs['train_args'].update(**out_paths)
    kwargs['train_args']['batch_size'] = kwargs['data_args']['batch_size']
    kwargs['train_args']['dim_z'] = kwargs['model_args']['dim_z']
    config(config_file=args.config_file, **kwargs)
    
    main(**kwargs)  
