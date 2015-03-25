from ..NetworkUtil import *
from theano.tensor.signal import downsample
import theano.tensor as T
from Layer import Layer
import numpy as np

class PoolLayer(Layer):

    def __init__(self, pool_shape):
        if len(pool_shape) != 2:
            raise "only 2D pool layers supported"
        self.pool_shape = pool_shape

    def feedforward_symb(self, x_symb, shape_in):
        pool_dims = len(self.pool_shape)
        if len(shape_in) != pool_dims +1 != 3:
            raise "pool can't be applied to input shape"
        n_features = shape_in[0]
        shape_out = [n_features, shape_in[-2] / self.pool_shape[-2], shape_in[-1] / self.pool_shape[-1]]
        y = downsample.max_pool_2d(x_symb, self.pool_shape, ignore_border=True)
        self.layer_shape = shape_out
        return y, shape_out

