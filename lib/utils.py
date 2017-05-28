'''Utilities

'''

import argparse
import logging
import os
from os import path
import random
import yaml

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from log_util import set_file_logger


trng = RandomStreams(random.randint(1, 1000000))
logger = logging.getLogger('BGAN.util')


try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1


def print_section(s):
    '''For printing sections to scripts nicely.
    Args:
        s (str): string of section
    '''
    h = s + ('-' * (_columns - len(s)))
    print h
    

def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help=('Output path directory. All model results will go'
                              ' here. If a new directory, a new one will be '
                              'created, as long as parent exists.'))
    parser.add_argument('-n', '--name', default=None,
                        help=('Name of the experiment. If given, base name of '
                              'output directory will be `--name`. If not given,'
                              ' name will be the base name of the `--out_path`')
                        )
    parser.add_argument('-a', '--architecture', default=None,
                        help=('Architecture name. This must be registered. '
                              'See `models/__init__.py` for details'))
    parser.add_argument('-S', '--source', type=str, default=None,
                        help='Dataset locations (full path).')
    parser.add_argument('-c', '--config_file', default=None,
                        help=('Configuration yaml file. '
                              'See `exps/` for examples'))
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser


def setup_out_dir(out_path, name=None):
    '''Sets up the output directory of an experiment.
    
    '''
    if out_path is None:
        raise ValueError('Please set `--out_path` (`-o`) argument.')
    if name is not None:
        out_path = path.join(out_path, name)
        
    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')
    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)
    if not path.isdir(binary_dir):
        os.mkdir(binary_dir)
    if not path.isdir(image_dir):
        os.mkdir(image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir)


def config(data_args=None, model_args=None, loss_args=None, optimizer_args=None,
           train_args=None, config_file=None):
    '''Loads arguments into a yaml file.
    
    '''
    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        if model_args is not None:
            model_args.update(**d.get('model_args', {}))
        if loss_args is not None:
            loss_args.update(**d.get('loss_args', {}))
        if optimizer_args is not None:
            optimizer_args.update(**d.get('optimizer_args', {}))
        if train_args is not None:
            train_args.update(**d.get('train_args', {}))
        if data_args is not None:
            data_args.update(**d.get('data_args', {}))
            
    
    logger.info('Training model with: \n\tdata args {}, \n\toptimizer args {} '
                '\n\tmodel args {} \n\tloss args {} \n\ttrain args {}'.format(
                    data_args, optimizer_args, model_args, loss_args,
                    train_args))
        

def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]