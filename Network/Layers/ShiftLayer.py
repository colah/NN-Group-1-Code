from ..NetworkUtil import *
import theano.tensor as T
from Layer import Layer
from theano import function, shared
import numpy as np

class ShiftLayer(Layer):

    def __init__(self, anchors):
        self.anchors = anchors
        self.shape_in = None

    def refresh_anchors(self, f):
        self.cs.set_value(f(self.anchors))

    def feedforward_symb(self, x_symb, shape_in):

        N = len(self.anchors)

        if not self.shape_in:
            self.shape_in = shape_in
            self.vs = self.new_param([N,] + shape_in) # [N, X]
            self.cs = shared(np.zeros([N,] + shape_in)) # [N, X]
        elif self.shape_in != shape_in:
            raise "Fully shift layer can't be \
                   reconnected to a previous layer of a \
                   different size."

        vs = self.vs
        xs =  x_symb.dimshuffle(0, 'x', 1)     # [B, N, X]
        cs = self.cs.dimshuffle('x', 0, 1)     # [B, N, X]
        ks = T.exp(-((xs-cs)**2/9).sum(2))       # [B, N]
        vs = vs.dimshuffle('x', 0,1)
        ks =ks.dimshuffle(0, 1, 'x')            # [B, N, X]
        kvs = (ks*vs).sum(1)
        denom = 1+T.sqrt(0.01+(kvs**2).sum(1))\
             .dimshuffle(0, 'x')          # [B, N, X]


        return x_symb + kvs/denom, self.shape_in

    def debugf(self):
        x_symb = T.matrix()

        vs = self.vs
        xs =  x_symb.dimshuffle(0, 'x', 1)     # [B, N, X]
        cs = self.cs.dimshuffle('x', 0, 1)     # [B, N, X]
        ks = T.exp(-((xs-cs)**2).sum(2))       # [B, N]
        vs = vs.dimshuffle('x', 0,1)
        ks =ks.dimshuffle(0, 1, 'x')            # [B, N, X]
	kvs = (ks*vs).sum(1)
        denom = T.sqrt((kvs**2).sum(1))\
             .dimshuffle(0, 'x')          # [B, N, X]


        return function([x_symb], denom)#[(x_symb + (ks*vs).sum(1)/denom)]) 


