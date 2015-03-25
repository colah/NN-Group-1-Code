# We're going to implement a sigmoid regression,
# a one layer neural network with sigmoid activaiton functions.

# Load MNIST
import data
mnist = data.load("mnist", label_format = "vector", flatvecs = True)

train_set = mnist["train"]
test_set = mnist["test"]

# ys are unit vectors which are 1 in the answer direction.
# For example:
#
# array([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
#
# Instead of [5 0].

# Theano is the library we're going to use
# to implement our neural networks.

import numpy as np

from theano import shared, function
import theano.tensor as T
from theano.tensor.nnet import sigmoid

# Refer to ex02 for more on Theano.

# Model:

x = T.matrix()

W = shared( 0.01 * np.random.randn(784, 10) )
b = shared( np.zeros(10) )

y = sigmoid( T.dot(x, W) + b)


# cost

target = T.matrix()

cost = T.mean((y-target)**2)

# Alterantively, you can use the following
# which adds some regularization
cost = T.mean((y-target)**2) + 0.0001*T.sum(W**2)

# Functions to use model:

feedforward = function([x], y)

k = T.scalar()

test = function([x, target], cost)

train = function([x, target, k], cost,
            updates = [(W, W - k*T.grad(cost, W)),
                       (b, b - k*T.grad(cost, b))]  )

# Train

for n in range(20):
    print "Test Cost:", test(test_set["xs"],test_set["ys"])
    for m in range(2000):
        batch_x, batch_y = train_set.get_batch(30)
        train(batch_x, batch_y, 0.1)

# We'd probably like to know what our resulting accuracy is.
# Don't worry too much about this code:

y_guesses = T.argmax(y, axis = 1)
target_ans =  T.argmax(target, axis = 1)

accuracy = T.mean(T.eq(y_guesses, target_ans))

test_accuracy = function([x, target], accuracy)

print ""
print "Test Accuracy:", 100*test_accuracy(test_set["xs"],test_set["ys"]), "%"

#Let's visualize the filers we learned...

import vis

W = W.get_value()

print ""
print "Filters Learned:"
for n in range(10):
    print n
    print ""
    vis.vis_matrix_signed(W[:, n].reshape(28,28))
    print "---------"
