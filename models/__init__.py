'''Main module for models.

This is for use of the `arch` argument in the `model_args`. If you have your
own model you wish to try, just follow the examples found in the parent
directory. If the `build_model` function found here doesn't suit your needs,
one can be added to the architecture module directly. This should be auto-
detected and used.

'''

import logging

import dcgan_28_pub, dcgan_32, dcgan_32_pub, dcgan_64, dcgan_64_pub


logger = logging.getLogger('BGAN.models')
_models = {}


def register(name, model_build):
    global _models
    
    _models[name] = model_build
    

def build_model(module, noise_var, input_var,
                dim_z, dim_h=128, use_batch_norm=True, leak=0.02,
                dim_x=None, dim_y=None, dim_c=None, nonlinearity=None):
    '''Example build function for discriminator and generator.
    
    .. note::
    
        This is a basic temmplate for model building. A custom one can be
        created in a custom architecture module.
    
    '''
    if dim_x is not None and dim_x != module.DIM_X:
        logger.warning('Changing x dim to {} from {}, may not work correctly '
                       'with this architecture.'.format(dim_x, module.DIM_X))
        module.DIM_X = dim_x
        
    if dim_y is not None and dim_y != module.DIM_Y:
        logger.warning('Changing x dim to {} from {}, may not work correctly '
                       'with this architecture.'.format(dim_y, module.DIM_Y))
        module.DIM_Y = dim_y
        
    if dim_c is not None and dim_c != module.DIM_C:
        module.DIM_C = dim_c
        
    if nonlinearity is not None and nonlinearity != module.NONLIN:
        logger.info('Setting nonlinearity of generator output to {}'.format(
            nonlinearity))
        if nonlinearity == 'identity': nonlinearity = None
        module.NONLIN = nonlinearity
    
    generator = module.build_generator(
        input_var=noise_var, dim_z=dim_z, dim_h=dim_h)
    discriminator = module.build_discriminator(
        input_var=input_var, dim_h=dim_h, use_batch_norm=use_batch_norm,
        leak=leak)
    
    return generator, discriminator
    

def build(noise_var, input_var, arch=None, **model_args):
    '''Builds the generator and discriminator.
    
    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.
    
    '''
    
    logger.info('Using architecture `{}`'.format(arch))
    module = _models.get(arch, None)
    if module is None:
        raise ValueError('Arch not found (``). Did you register it? '
                         'Available: {}'.format(
            arch, _models.keys()))

    if hasattr(module, 'build_model'):
        logger.debug('Using custom `build` found within module.')
        return getattr(module, 'build_model')(module, noise_var, input_var,
                                       **model_args)
    else:
        return build_model(module, noise_var, input_var, **model_args)
    

register('dcgan_28_pub', dcgan_28_pub)
register('dcgan_32', dcgan_32)
register('dcgan_32_pub', dcgan_32_pub)
register('dcgan_64', dcgan_64)
register('dcgan_64_pub', dcgan_64_pub)
