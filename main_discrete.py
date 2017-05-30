'''Main function to train discrete BGAN.

'''

import logging

import lasagne
import numpy as np
import theano
import theano.tensor as T

from lib.data import load_stream
from lib.log_util import set_stream_logger
from lib.loss import get_losses_discrete
from lib.math import est_log_Z
from lib.train import setup, train
from lib.utils import config, make_argument_parser, print_section, setup_out_dir
from lib.viz import setup as setup_viz
from models import build


logger = logging.getLogger('BGAN')


def main(data_args=None, optimizer_args=None, model_args=None, loss_args=None,
         train_args=None):
    '''Main function for discrete BGAN.
    
    '''
      
    print_section('LOADING DATA') ##############################################
    train_stream, training_samples, shape, viz_options = load_stream(
        **data_args)
    train_args['training_samples'] = training_samples
    setup_viz(**viz_options)
    model_args.update(**shape)
    loss_args.update(**shape)
    loss_args['batch_size'] = data_args['batch_size']
    
    print_section('MODEL') #####################################################
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    
    log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')
    loss_args['loss_options'] = loss_args.get('loss_options', None) or {}
    loss_args['loss_options']['log_Z'] = log_Z

    logger.info('Building model and compiling GAN functions...')
    logger.info('Model args: {}'.format(model_args))
    generator, discriminator = build(noise_var, input_var, **model_args)

    g_output_logit = lasagne.layers.get_output(generator)
    
    g_results, d_results, log_Z_est = get_losses_discrete(
        discriminator, g_output_logit, optimizer_args=optimizer_args,
        **loss_args)
    
    g_results.update(**{
        'log Z': log_Z,
        'log Z (est)': log_Z_est.mean()
    })
        
    print_section('OPTIMIZER') #################################################
    train_d, train_g, gen =  setup(input_var, noise_var, log_Z, generator,
                                   discriminator, g_results, d_results,
                                   discrete=True, **optimizer_args)
        
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
        discrete=True,
        downsample_to=(32, 32)
    ),
    optimizer_args=dict(
        optimizer='adam',
        optimizer_options=dict(beta1=0.5),
        learning_rate=1e-4,
    ),
    model_args=dict(
        arch='dcgan_28_pub',
        dim_z=64,
        dim_h=64,
        leak=0.2,
        use_batch_norm=False
    ),
    loss_args=dict(
        loss='binary_bgan',
        n_samples=20
    ),
    train_args=dict(
        epochs=100,
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
    config(config_file=args.config_file, **kwargs)
    kwargs['train_args']['batch_size'] = kwargs['data_args']['batch_size']
    kwargs['train_args']['dim_z'] = kwargs['model_args']['dim_z']
    
    main(**kwargs)  
