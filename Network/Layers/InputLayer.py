from ..NetworkUtil import *
from Layer import Layer
import numpy as np

class InputLayer(Layer):

    def __init__(self, layer_shape):
        if isinstance(layer_shape, int):
            layer_shape = [layer_shape]
        self.layer_shape = layer_shape
        self.symb_input = tensor(1+len(layer_shape))

    def feedforward_symb(self, x, shape):
        return self.symb_input, self.layer_shape

