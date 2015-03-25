from ..NetworkUtil import *
import theano.tensor as T
from theano.tensor.nnet import softmax
from Layer import Layer
import numpy as np

class SoftMaxLayer(Layer):


    def __init__(self, layer_shape, init_wts_sd = None, init_bias=0.0):
        if isinstance(layer_shape, int):
            layer_shape = [layer_shape]
        self.layer_shape = layer_shape
        assert len(layer_shape) == 1
        self.shape_in = None
        self.init_wts_sd = init_wts_sd
        self.init_bias = init_bias

    def feedforward_symb(self, x_symb, shape_in):
        if not self.shape_in:
            size_in    = product(shape_in)
            size_layer = product(self.layer_shape)
            self.shape_in = shape_in
            self.W = self.new_param(
                (size_in, size_layer), s = self.init_wts_sd or product(shape_in)**(-0.5))
            self.b = self.new_param((size_layer,), initial=self.init_bias)
        elif self.shape_in != shape_in:
            raise "Fully connected layer can't be \
                   reconnected to a previous layer of a \
                   different size."
        else:
            size_in    = product(shape_in)
            size_layer = product(self.layer_shape)
        if len(shape_in) != 1:
            x_symb = x_symb.reshape((x_symb.shape[0], size_in))
        y = softmax( T.dot(x_symb, self.W) - self.b)
        return y, self.layer_shape



