'''Train function.

'''

import logging
from os import path
import time

import lasagne
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano
from theano import tensor as T

from utils import update_dict_of_lists
from viz import save_images, save_movie


logger = logging.getLogger('BGAN.train')
floatX_ = lasagne.utils.floatX
floatX = theano.config.floatX

GENERATOR = None
DISCRIMINATOR = None


def summarize(summary, gen_fn, dim_z, prefix='', image_dir=None):
    if len(prefix) > 0: prefix = prefix + '_'
    logger.info(summary)
    samples = gen_fn(floatX_(np.random.rand(5000, dim_z)))[0:64]
    if image_dir is not None:
        logger.debug('Saving images to {}'.format(image_dir))
        out_path = path.join(image_dir, '{}gen.png'.format(prefix))
        save_images(samples, 8, 8, out_file=out_path)
        

def make_gif(gen_fn, z=None, samples=[], prefix='', image_dir=None):
    if len(prefix) > 0: prefix = prefix + '_'
    samples.append(gen_fn(floatX_(z))[0:64])
    out_path = path.join(image_dir, '{}movie.gif'.format(prefix))
    save_movie(samples, 8, 8, out_file=out_path)
        

def save(results, prefix='', binary_dir=None):
    if len(prefix) > 0: prefix = prefix + '_'
    np.savez(path.join(binary_dir, '{}gen_params.npz'.format(prefix)),
             *lasagne.layers.get_all_param_values(GENERATOR))
    np.savez(path.join(binary_dir, '{}disc_params.npz'.format(prefix)),
             *lasagne.layers.get_all_param_values(DISCRIMINATOR))
    np.savez(path.join(binary_dir, '{}results.npz'.format(prefix)),
             **results)
    
    
def setup(input_var, noise_var, log_Z,
          generator, discriminator, g_results, d_results, discrete=False,
          log_Z_gamma=None, clip=None, optimizer=None, optimizer_options=None):
    global GENERATOR, DISCRIMINATOR
    GENERATOR = generator
    DISCRIMINATOR = discriminator
    
    generator_loss = g_results.get('g loss', None)
    if generator_loss is None:
        raise ValueError('Generator loss not found in results.')

    discriminator_loss = d_results.get('d loss', None)
    if discriminator_loss is None:
        raise ValueError('Discriminator loss not found in results.')

    generator_params = lasagne.layers.get_all_params(
        GENERATOR, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(
        DISCRIMINATOR, trainable=True)

    logger.info('Training with {} and optimizer options {}'.format(
        optimizer, optimizer_options))
    
    if callable(optimizer):
        op = optimizer
    elif optimizer == 'adam':
        op = lasagne.updates.adam
    elif optimizer == 'rmsprop':
        op = lasagne.updates.rmsprop
    else:
        raise NotImplementedError('Optimizer not supported `{}`'.format(
            optimizer))
    generator_updates = op(
        generator_loss, generator_params, **optimizer_options)
    discriminator_updates = op(
        discriminator_loss, discriminator_params, **optimizer_options)
    
    if clip is not None:
        logger.info('Clipping weights with clip of {}'.format(CLIP))
        for k in discriminator_updates.keys():
            if k.name == 'W':
                discriminator_updates[k] = T.clip(
                    discriminator_updates[k], -clip, clip)

    if 'log Z (est)' in g_results.keys():
        logger.info('Updating log Z estimate with gamma {}'.format(log_Z_gamma))
        generator_updates.update(
            [(log_Z, log_Z_gamma * log_Z
              + (1. - log_Z_gamma) * g_results['log Z (est)'])])

    # COMPILE
    logger.info('Compiling functions.')
    train_discriminator = theano.function([noise_var, input_var],
        d_results, allow_input_downcast=True, updates=discriminator_updates)

    train_generator = theano.function([noise_var],
        g_results, allow_input_downcast=True, updates=generator_updates)

    gen_out = lasagne.layers.get_output(generator, deterministic=True)
    if discrete:
        if generator.output_shape[1] == 1:
            print "True"
            gen_out = T.nnet.sigmoid(gen_out)
            
    gen_fn = theano.function(
        [noise_var], gen_out)
    
    return train_discriminator, train_generator, gen_fn


def train(train_d, train_g, gen, stream, 
          summary_updates=None, epochs=None, training_samples=None,
          num_iter_gen=None, num_iter_disc=None, batch_size=None, dim_z=None,
          image_dir=None, binary_dir=None, archive_every=None):
    '''Main train function.
    
    '''

    # TRAIN
    logger.info('Starting training of GAN...')
    total_results = {}
    exp_name = binary_dir.split('/')[-2]
    rep_samples = floatX_(np.random.rand(5000, dim_z))
    
    train_sample = (stream.get_epoch_iterator(as_dict=True)
                    .next()['features'][:64])
    logger.debug('Saving images to {}'.format(image_dir))
    out_path = path.join(image_dir, 'training_example.png')
    save_images(train_sample, 8, 8, out_file=out_path)
    
    for epoch in range(epochs):
        u = 0
        start_time = time.time()
        
        results = {}
        widgets = ['Epoch {} ({}), '.format(epoch, exp_name), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets,
            maxval=(training_samples // batch_size)).start()
        
        for batch in stream.get_epoch_iterator(as_dict=True):
            inputs = batch['features']
            if inputs.shape[0] == batch_size:
    
                for i in range(num_iter_disc):
                    noise = floatX_(np.random.rand(len(inputs), dim_z))
                    d_outs = train_d(noise, inputs)
                    d_outs =  dict((k, np.asarray(v))
                        for k, v in d_outs.items())
                    update_dict_of_lists(results, **d_outs)
    
                for i in range(num_iter_gen):
                    noise = floatX_(np.random.rand(len(inputs), dim_z))
                    g_outs = train_g(noise)
                    g_outs = dict((k, np.asarray(v)) for k, v in g_outs.items())
                    update_dict_of_lists(results, **g_outs)
    
                u += 1
                pbar.update(u)
                if summary_updates is not None and u % summary_updates == 0:
                    summary = dict((k, np.mean(v)) for k, v in results.items())
                    summarize(summary, gen, dim_z, prefix=exp_name,
                              image_dir=image_dir)
                    
        logger.info('Total Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, epochs, time.time() - start_time))
        
        if archive_every is not None and epoch % archive_every == 0:
            prefix = '{}({})'.format(exp_name, epoch)
        else:
            prefix = exp_name
        
        result_summary = dict((k, np.mean(v)) for k, v in results.items())
        update_dict_of_lists(total_results, **result_summary)
        summarize(result_summary, gen, dim_z, prefix=prefix,
                  image_dir=image_dir)
        make_gif(gen, z=rep_samples, prefix=exp_name, image_dir=image_dir)
        save(total_results, prefix=prefix, binary_dir=binary_dir)