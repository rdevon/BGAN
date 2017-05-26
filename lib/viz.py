'''Visualization.

'''

import logging

import imageio
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
from PIL import Image


logger = logging.getLogger('BGAN.viz')

_options = dict(
    use_tanh=False,
    quantized=False,
    img=None
)


def setup(use_tanh=None, quantized=None, img=None):
    global _options
    if use_tanh is not None:
        _options['use_tanh'] = use_tanh
    if quantized is not None:
        _options['quantized'] = quantized
    if img is not None:
        _options['img'] = img


def dequantize(images):
    images = np.argmax(images, axis=1).astype('uint8')
    images_ = []
    for image in images:
        img2 = Image.fromarray(image)
        img2.putpalette(_options['img'].getpalette())
        img2 = img2.convert('RGB')
        images_.append(np.array(img2))
    images = np.array(images_).transpose(0, 3, 1, 2)
    return images


def save_images(images, num_x, num_y, out_file=None):
    if out_file is None:
        logger.warning('`out_file` not provided. Not saving.')
    else:
        
        if _options['quantized']:
            images = dequantize(images)
        
        elif _options['use_tanh']:
            images = 0.5 * (images + 1.)


        dim_c, dim_x, dim_y = images.shape[-3:]
        if dim_c == 1:
            plt.imsave(out_file,
                       (images.reshape(num_x, num_y, dim_x, dim_y)
                        .transpose(0, 2, 1, 3)
                        .reshape(num_x * dim_x, num_y * dim_y)),
                       cmap='gray')
            
        else:            
            scipy.misc.imsave(
                out_file, (images.reshape(num_x, num_y, dim_c, dim_x, dim_y)
                           .transpose(0, 3, 1, 4, 2)
                           .reshape(num_x * dim_x, num_y * dim_y, dim_c)))
        
        
        
def save_movie(images, num_x, num_y, out_file=None):
    if out_file is None:
        logger.warning('`out_file` not provided. Not saving.')
    else:
        images_ = []
        for i, image in enumerate(images):
            if _options['quantized']:
                image = dequantize(image)
            dim_c, dim_x, dim_y = image.shape[-3:]
            image = image.reshape((num_x, num_y, dim_c, dim_x, dim_y))
            image = image.transpose(0, 3, 1, 4, 2)
            image = image.reshape(num_x * dim_x, num_y * dim_y, dim_c)
            if _options['use_tanh']:
                image = 0.5 * (image + 1.)
            images_.append(image)
        imageio.mimsave(out_file, images_)