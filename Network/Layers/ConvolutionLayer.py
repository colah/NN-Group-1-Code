from ..NetworkUtil import *
from theano.tensor.nnet import conv
import theano.tensor as T
from Layer import Layer
import numpy as np

class ConvolutionLayer(Layer):

    def __init__(self, conv_shape, n_features, init_wts_sd, 
                 init_bias=0.0, activation = None):
        if len(conv_shape) != 2:
            raise "only 2D conv layers supported"
        self.conv_shape = conv_shape
        self.n_features = n_features
        self.init_wts_sd = init_wts_sd
        self.init_bias = init_bias
        self.activation = activation
        self.W = None

    def feedforward_symb(self, x_symb, shape_in):
        conv_dims = len(self.conv_shape)
        if len(shape_in) < conv_dims:
            raise "convolution can't be applied to input shape"
        elif len(shape_in) == 3:
            W_shape = [self.n_features] + shape_in[:1] + self.conv_shape
        elif len(shape_in) == 2:
            x_symb = x_symb.dimshuffle(0, 'x', 1, 2)
            W_shape = [self.n_features] + [1] + self.conv_shape
        shape_out = [self.n_features, shape_in[-2] - self.conv_shape[0] + 1, shape_in[-1] - self.conv_shape[1] + 1]
        size_out = product(shape_out)

        if not self.W:
            self.W = self.new_param(W_shape, s = self.init_wts_sd)
            self.b = self.new_param((self.n_features,), initial=self.init_bias)

        conv_out = conv.conv2d(x_symb, self.W, filter_shape = W_shape)
        y = self.activation(conv_out - self.b.dimshuffle(0, 'x', 'x') )
        self.layer_shape = shape_out
        return y, shape_out

