from ..NetworkUtil import *
from theano import function, shared
import theano
import numpy as np

class AugmentedParameter():
    def __init__(self, var, shape):
        self.var = var
        self.shape = shape
        self.momentum = shared(np.zeros(shape).astype(theano.config.floatX))
        self.base = shared(np.zeros(shape).astype(theano.config.floatX))
        self.size = product(self.shape)

class Layer:
    """ An abstract class for a single neural net layer.
    """

    # Note: shared by all instances until overridden.
    # We could set these in an __init__() but these
    # are kind of ugly since they need to be explicitly
    # called from subclasses.
    params = None
    param_shapes = None
    aug_params = []
    activation = None

    def feedforward_symb(self, x_symb, shape_in):
        """ This is the essential function for any
            layer to implement.

            It turns symbolic input activations into
            symbolic output activations, and input shape
            into output shape.

            It should also probably instantiate any paramters.

            A really simple example might look like:

            W = self.new_param((10, size_in), s = 1/3)
            b = self.new_param((10,), s = 1/3)
            return self.activation( T.dot(W, x_symb) - b), 10
        """
        pass

    def new_param(self, shape = None, s = 0, initial = None):
        """Create a parameter of shape `shape` and random
           initialization (variance `s`) or given `intial` values.

           These parameters are tensors that control the behavior
           of the layer, and should be subject to the learning algorithm.

           The learning algorithm can access the paramters through
           `layer.params` and useful shape information from
           `layer.param_shapes`.

           The parameters are Theano shared variables.

           `initial` can be either:

           (1) None: Initialized by a Gaussian random variable with
           mean 0 and standard deviation `s`.

           (2) A float: Initialized to a constant across the whole
           shape.

           (3) An array of floats: In this case the shape should be
           `None` (and is otherwise ignored), and the array is used to
           do the initialization.
        """
        if shape and not initial:
            # Create a numpy array with initial values.
            a = s * np.random.randn(*shape)
        elif initial and type(initial) == float:
            a = initial*np.ones(shape)
        elif initial:
            a = initial
            shape = a.shape
        else:
            raise "new_param must recieve shape or initial array."
        # Create a shared variable, forcing floats into the preferred type
        # (eg. float32 for on the GPU, float64 for CPU)
        a = shared(a.astype(theano.config.floatX))
        # Fix sharing
        if not self.params:
            self.params = []
            self.param_shapes = []
            self.aug_params = []
        self.params.append(a)
        self.param_shapes.append(shape)
        self.aug_params.append(AugmentedParameter(a, shape))
        return a

