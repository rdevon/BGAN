'''Data streams.

'''

import logging

import cv2
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
import h5py
import numpy as np
from PIL import Image
import theano


logger = logging.getLogger('BGAN.data')
floatX = theano.config.floatX


class Rescale(Transformer):
    def __init__(self, data_stream, min=0, max=1, use_tanh=True, **kwargs):
        super(Rescale, self).__init__(data_stream=data_stream,
                                      produces_examples=False, **kwargs)
        self.min = min
        self.max = max
        self.use_tanh = use_tanh
    
    def transform_batch(self, batch):
        index = self.sources.index('features')
        x = batch[index]
        x = float(self.max - self.min) * (x / 255.) - self.min
        if self.use_tanh: x = 2. * x - 1. 
        x = x.astype(floatX)
        batch = list(batch)
        batch[index] = x
        return tuple(batch)
    
    
class Quantize(Transformer):
    def __init__(self, data_stream, img, n_colors=16, downsample_to=None,
                 **kwargs):
        super(Quantize, self).__init__(data_stream=data_stream,
                                       produces_examples=False, **kwargs)
        self.img = img
        self.downsample_to = downsample_to
        self.n_colors = n_colors
    
    def transform_batch(self, batch):
        batch = list(batch)
        new_arr = []
        for arr in batch[0]:
            arr = arr.transpose(1, 2, 0)
            if self.downsample_to is not None:
                dim_x, dim_y = self.downsample_to
                arr = cv2.resize(arr, (dim_x, dim_y),
                                 interpolation=cv2.INTER_AREA)
            
            dim_x, dim_y = arr.shape[:2]
            
            img = Image.fromarray(arr)
            img = img.quantize(palette=self.img, colors=self.n_colors)
            arr = np.array(img)
            arr[arr > (self.n_colors - 1)] = 0 #HACK don't know why it's giving more colors sometimes
            
            arr_ = np.zeros((self.n_colors, dim_x * dim_y)).astype(arr.dtype)
            arr_[arr.flatten(), np.arange(dim_x * dim_y)] = 1
            new_arr.append(arr_.reshape((self.n_colors, dim_x, dim_y)))
        batch[0] = np.array(new_arr)
        batch = tuple(batch)
        return batch
    
    
class OneHotEncoding(Transformer):
    def __init__(self, data_stream, num_classes, **kwargs):
        super(OneHotEncoding, self).__init__(data_stream,
                                             produces_examples=False, **kwargs)
        self.num_classes = num_classes

    def transform_batch(self, batch):
        index = self.sources.index('features')
        x = batch[index]
            
        if np.max(x) >= self.num_classes:
            raise ValueError("all entries in source_batch must be lower than "
                             "num_classes ({})".format(self.num_classes))
        x = x.transpose(0, 2, 3, 1)
        shape = x.shape
        x = x.flatten()
        output = np.zeros((shape[0] * shape[1] * shape[2], self.num_classes),
            dtype=floatX)
        for i in range(self.num_classes):
            output[x == i, i] = 1
        output = output.reshape(
            (shape[0], shape[1], shape[2], self.num_classes))
        output = output.transpose(0, 3, 1, 2)
            
        batch = list(batch)
        batch[index] = output
        return tuple(batch)

    

def load_stream(batch_size=None, source=None, use_tanh=False, discrete=False,
                n_colors=16, downsample_to=None):
    if source is None:
        raise ValueError('Source not provided.')
    if batch_size is None:
        raise ValueError('Batch size not provided.')
    logger.info('Loading data from `{}`'.format(source))
    
    train_data = H5PYDataset(source, which_sets=('train',))
    num_train = train_data.num_examples
    logger.debug('Number of training examples: {}'.format(num_train))
    
    scheme = ShuffledScheme(examples=num_train, batch_size=batch_size)
    stream = DataStream(train_data, iteration_scheme=scheme)
    
    viz_options = dict(use_tanh=use_tanh)
    
    data = stream.get_epoch_iterator(as_dict=True).next()['features']
    if discrete:
        n_uniques = len(np.unique(data))
        if n_uniques == 256 and np.max(data) == 255:
            features = H5PYDataset(source, which_sets=('train',),
                                   subset=slice(0, 1000), sources=['features'])
            handle = features.open()
            arr, = features.get_data(handle, slice(0, 1000))
            arr = arr.transpose(0, 2, 3, 1)
            arr = arr.reshape((-1, arr.shape[2], arr.shape[3]))
            img = Image.fromarray(arr).convert(
                'P', palette=Image.ADAPTIVE, colors=n_colors)
            viz_options['img'] = img
            logger.info('Quantizing data to {} colors'.format(n_colors))
            viz_options['quantized'] = True
            if downsample_to is not None:
                logger.info('Downsampling the data to {}'.format(downsample_to))
            stream = Quantize(img=img, n_colors=n_colors,
                              downsample_to=downsample_to, data_stream=stream)
        else:
            logger.warning('Data already appears to be discretized. Skipping.')
    else:
        if np.max(data) != 255:
            raise ValueError('Data needs to be RGB, this doesn\'t appear to be '
                             'the case')
        stream = Rescale(stream, use_tanh=use_tanh)

    data = stream.get_epoch_iterator(as_dict=True).next()['features']
    shape = data.shape
    shape = dict(dim_c=shape[1], dim_x=shape[2], dim_y=shape[3])
        
    return stream, num_train, shape, viz_options