from theano import function, shared
import theano
import theano.tensor as T
import numpy as np

from NetworkUtil import *
from Layers import *

theano.config.exception_verbosity = "high"


class Network:
    def __init__(self, layers, activation = sigmoid, cost = cross_entropy):
        self.layers = layers
        self.cost = cost
        for layer in self.layers:
            if not layer.activation:
                layer.activation = activation
        self._build_symbolic()
        self._build_functions()

    def _build_symbolic(self):
        """ Build a symbolic model of the entire network.
        """
        self.symb_input  = self.layers[0].symb_input
        y, shape = None, None
        self.params, self.param_shapes, self.layer_ys = [], [], []
        for layer in self.layers:
            y, shape = layer.feedforward_symb(y, shape)
            self.layer_ys += [y]
            if layer.params:
                self.params += layer.params
                self.param_shapes += layer.param_shapes
        self.symb_output = y

    def _build_functions(self):
        """ Create Theano functions that underly higher level functionality.

            None of the created functions should be used directly by the user.
        """
        if self.cost == log_likelihood:
            target = T.lvector()
        else:
            target = tensor(1+len(self.layers[-1].layer_shape))
        if self.cost == log_likelihood:
            cost = -T.sum(T.log(self.symb_output)[T.arange(target.shape[0]), target])
        elif self.cost == mse:
            cost = T.sum((self.symb_output-target)**2)
        else:
            raise "unsupported cost function"
        # Feedforward an input.
        self._feedforward_fs = []
        for y in self.layer_ys:
            self._feedforward_fs += [function([self.symb_input], y)]
        self._feedforward = function([self.symb_input], self.symb_output)

        #Introspection!
        self.layer_infos = []
        to_compute = []
        def add_compute(p):
            to_compute.append(p)
            n = add_compute.n
            add_compute.n += 1
            return n
        add_compute.n = 0
        for layer, y, n in zip(self.layers, self.layer_ys, range(100)):
            param_name_count = 0
            def get_param_name(param):
                if param == layer.W: return "W"
                if param == layer.b: return "b"
                name = "param_" + str(param_name_count)
                return name
            info = { "name" : str(n) + "_" + layer.__class__.__name__ ,
                     "compute_names" : ["y", "y_grad"],
                     "compute_ns" : [add_compute(y), add_compute(T.grad(cost, y))] }
            if layer.activation and layer.activation == sigmoid:
                info["activation"] = "sigmoid"
            elif layer.activation and layer.activation == ReLU:
                info["activation"] = "ReLU"
            elif layer.activation and layer.activation == linear:
                info["activation"] = "linear"
            else:
                info["activation"] = None


            for p in layer.params or []:
                p_name = get_param_name(p)
                info["compute_names"].append( p_name )
                info["compute_ns"].append( add_compute(p) )
                info["compute_names"].append( p_name + "_grad" )
                info["compute_ns"].append( add_compute( T.grad(cost, p) ) )
            self.layer_infos.append(info)
        self._complete_introspect = function(
            [self.symb_input, target], to_compute) 
        
        #Test performance on some input vs target answer
        self._test_cost = function(
            [self.symb_input, target], cost)
        aug_params = [x for l in self.layers for x in l.aug_params]

        # Scale parameter momentum
        scale_constant = T.scalar()
        self._scale_param_momentum = function([scale_constant], [],
            updates = [ (x.momentum, scale_constant*x.momentum ) for x in aug_params ]
            )

        # Weight decay
        decay_constant = T.scalar()
        self._scale_weights = function([decay_constant], [],
            updates = [ (x.var, decay_constant*x.var ) for x in aug_params ]
            )
        idscale_constant = T.scalar()
        self._scale_weights_relId = function([decay_constant, idscale_constant], [],
            updates = [ (x.var, idscale_constant*T.identity_like(x.var) + decay_constant*(x.var-idscale_constant*T.identity_like(x.var)) ) 
                        for l in self.layers[:-1] for x in l.aug_params[:1] ]
            )
        self._add_weights_kId = function([idscale_constant], [],
            updates = [ (x.var, x.var + idscale_constant*T.identity_like(x.var) ) 
                        for l in self.layers[:-1] for x in l.aug_params[:1] ]
            )

        # Update momentum based on cost gradient for given learning rate, input and target.
        learning_rate = T.scalar()
        self._momentum_deriv_add = function([self.symb_input, target, learning_rate], [],
            updates = [(x.momentum, x.momentum - learning_rate*T.grad(cost, x.var)) for x in aug_params]
            )
        learning_rates = [T.scalar() if l.params else None for l in self.layers ]
        used_learning_rates = filter(lambda x: x != None, learning_rates)
        self._momentum_deriv_add_perlayer = function([self.symb_input, target] + used_learning_rates, [],
            updates = [(x.momentum, x.momentum - learning_rate_*T.grad(cost, x.var)) 
                for learning_rate_, l in zip(learning_rates, self.layers) for x in l.aug_params]
            )
        # NORMALIZED SGD: Update momentum based on cost gradient for given learning rate, input and target.
        self._momentum_deriv_add_normalized = function([self.symb_input, target, learning_rate], [],
            updates = [(x.momentum, x.momentum - learning_rate*T.sqrt(float(x.size))*norm_L2(T.grad(cost, x.var)) ) for x in aug_params]
            )
        #print [product(x.shape) for x in aug_params]
        self._momentum_deriv_add_perlayer_normalized = function([self.symb_input, target] + used_learning_rates, [],
            updates = [(x.momentum, x.momentum - learning_rate*T.sqrt(float(x.size))*norm_L2(T.grad(cost, x.var)) ) 
                for learning_rate, l in zip(learning_rates, self.layers) for x in l.aug_params]
            )
        # Update parameters based on paramter momentum
        self._learn = function([], [],
            updates = [(x.var, x.var + x.momentum) for x in aug_params ]
            )
        # Nesterov method
        self._nesterov_reset_base = function([], [],
            updates = [(x.base, x.var) for x in aug_params]
            )
        self._nesterov_set_params = function([], [],
            updates = [(x.var, x.base + x.momentum) for x in aug_params]
            )
        self._nesterov_learn = function([], [],
            updates = [(x.base, x.base + x.momentum) for x in aug_params]
            )

        # accuracy!
        if self.cost == log_likelihood:
            guesses = T.argmax(self.symb_output, axis = 1)
            self._corrects = theano.function([self.symb_input, target], T.sum(T.eq(guesses, target)))
            
            self._av_correct_confidence = function(
                [self.symb_input, target],
                T.mean(self.symb_output[T.arange(target.shape[0]), target])
                )

            self._inspect_grad = function([self.symb_input, target], 
                [T.grad(cost, x.var) for x in aug_params]
                )

    def introspect(self, xs, ys):
        introspection = self._complete_introspect(xs, ys)
        human_readable = {}
        for layer_info in self.layer_infos:
            activity = {}
            for name, n in zip(layer_info["compute_names"], layer_info["compute_ns"]):
                activity[name] = introspection[n]
            human_readable[layer_info["name"]] = activity
        return human_readable

    def train(self, xs, ys, learning_rate = 0.01, learning_rates = None, normalize = False):
        self._scale_param_momentum(0.0)
        if learning_rates:
            if normalize:
                self._momentum_deriv_add_perlayer_normalized(xs, ys, *learning_rates)
            else:
                self._momentum_deriv_add_perlayer(xs, ys, *learning_rates)
        else:
            if normalize:
                self._momentum_deriv_add_normalized(xs, ys, learning_rate)
            else:
                self._momentum_deriv_add(xs, ys, learning_rate)
        self._learn()

    def train2(self, xs, ys, xs2, ys2, learning_rate = 0.01, learning_rates = None, normalize = False):
        self._scale_param_momentum(0.0)
        if learning_rates:
            if normalize:
                self._momentum_deriv_add_perlayer_normalized(xs, ys, xs2, ys2, *learning_rates)
            else:
                self._momentum_deriv_add_perlayer(xs, ys, xs2, ys2, *learning_rates)
        else:
            if normalize:
                self._momentum_deriv_add_normalized(xs, ys, xs2, ys2, learning_rate)
            else:
                self._momentum_deriv_add(xs, ys, xs2, ys2, learning_rate)
        self._learn()

    def train_momentum(self, xs, ys, learning_rate = 0.01, friction = 0.05, learning_rates = None, normalize = False):
        k = 1.0 - friction
        self._scale_param_momentum(k)
        if learning_rates:
            if normalize:
                self._momentum_deriv_add_perlayer_normalized(xs, ys, *learning_rates)
            else:
                self._momentum_deriv_add_perlayer(xs, ys, *learning_rates)
        else:
            if normalize:
                self._momentum_deriv_add_normalized(xs, ys, learning_rate)
            else:
                self._momentum_deriv_add(xs, ys, learning_rate)
        self._learn()

    def train_nesterov(self, xs, ys, learning_rate = 0.01, friction = 0.05):
        k = 1.0 - friction
        self._scale_param_momentum(k)
        self._nesterov_set_params()
        self._momentum_deriv_add(xs, ys, learning_rate)
        self._nesterov_learn()

    def test(self, xs, ys):
        """Evaluate cost function `test_data.
        """
        return self._test_cost(xs,ys) / len(ys) / product(self.layers[-1].layer_shape)

    def accuracy(self, xs, ys):
        """Give classification accuracy on test_data, assuming a classifcation problem.

           This function give the fraction of the test_data for which the dimension 
           in which y has the largest value is the dimension in which the activation
           has the largest value.
        """
        return self._corrects(xs, ys)/float(len(xs))

    def stochastic_accuracy(self, xs, ys):
        """Give average classification accuracy on test_data, assuming a classifcation problem,
           assuming output is a probability distribution over answers.
        """
        return self._av_correct_confidence(xs, ys)


    def __call__(self, x):
        """Feedforward operations on a neural net are effectively evaluation..."""
        x = np.asarray(x).astype(theano.config.floatX)
        layer0_shape = list(self.layers[0].layer_shape)
        x_shape = list(x.shape)
        if len(x_shape) == len(layer0_shape) - 1:
            x.reshape(-1, *x_shape)
        if x_shape[1:] != layer0_shape:
            raise Exception(
                """Neural net input must be the same size as input layer.
                Net Input: {0}
                Input Layer: {1}
                """.format(x_shape[1:], layer0_shape))
        return self._feedforward(x)

    def get_state(self):
        aug_params = [x for l in self.layers for x in l.aug_params]
        return [x.var.get_value() for x in aug_params]

    def set_state(self, state):
        aug_params = [x for l in self.layers for x in l.aug_params]
        for x, val in zip(aug_params, state):
            x.var.set_value(val)

