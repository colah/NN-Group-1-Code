# Theano works by you describing what you want to do
# with symbolic variables. You describe the opeartion
# you want once on the symbolic variables, and then 
# it compiles code to do that.
#
# Chris will talk about this, but if you need a refrence:
# http://deeplearning.net/software/theano/tutorial/

# Import Theano
from theano import shared, function
import theano.tensor as T

# Make two symbolic variables
a = T.vector()
b = T.vector()

# create an expression with them
c = a + 2*b

# compile a function
f = function([a,b], c)

# use it
print f([1,2], [0,3])

# Result:
# [ 1.  8.]
