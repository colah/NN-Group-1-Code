import theano.tensor as T
import theano
import theano.tensor.nnet as TNN

#### Neuron cost functions
def cross_entropy(a, y):
    return T.sum(-y*T.log(a)-(1-y)*T.log(1-a))

def mse(a, y):
    return T.sum((a-y)**2)

def log_likelihood(a, y_index):
    """Returns the negative log likelihood.  It has a different
    interface than other cost functions because `y` is no longer
    naturally a vector, but rather is a label which is to be used as
    an index on the output activation."""
    return -T.log(a[y_index])

#### Neuron activation functions

def linear(x):
    """Symbolically apply the identity function."""
    return x

def ReLU(x):
    """Symbolically apply the function x -> max(0, x)."""
    return T.maximum(0, x)

def BReLU(x):
    """Symbolically apply the BReLU function, x -> exp(x) if x < 1 else x."""
    return T.switch(x<1, T.exp(x), x)

def BReLU2(x):
    """Symbolically apply the BReLU function, x -> exp(x-1) if x < 1 else x."""
    return T.switch(x<1, T.exp(x-1), x)

def sigmoid(x):
    """Symbolically apply the sigmoid function."""
    return 1 / (1 + T.exp(-x))

def LogMagnitude(x):
    """Symbolically apply x -> log(x+1) if x > 0 else -log(1-x)."""
    return T.switch(x>0, T.log(x+1), -T.log(1-x))

def SqrtMagnitude(x):
    """Symbolically apply x -> log(x+1) if x > 0 else -log(1-x)."""
    return T.switch(x>0, T.sqrt(x), -T.sqrt(-x))

tanh = T.tanh

softplus = TNN.softplus


#### Utility functions for Theano
def product(l):
    """Multiply elements of a `list`. Like sum but for multiplication. """
    ret = 1
    if l:
        for x in l: ret *= x
    return ret

def tensor(n):
    """Create an n-dimensional Theano symbolic tensor. """
    return T.TensorType(theano.config.floatX, (False,)*n)()


def norm_L2(x):
    return x/T.sqrt(T.sum(x**2))

