#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# Example 05 -- An MNIST Autoencoder
#

import math
from Network import *
import data
import numpy
import vis


mnist = data.load("mnist", flatvecs = True)
test_data_xs = mnist["test"]["xs"]
train_data = mnist["train"]

net = Network(
    layers = [InputLayer([784]),
              FullyConnectedLayer([20], activation = sigmoid),
              FullyConnectedLayer([784], activation = sigmoid)],
    cost=mse)

for i in range(30):

    for m in range(150):
        batch_xs, _ = train_data.get_batch(30)
        net.train(batch_xs, batch_xs, learning_rate = 0.01 )

    print i, "Test Cost:", net.test(test_data_xs, test_data_xs )
    # Let's look at how it reproduces a few examples...
    examples = net(test_data_xs[0:3])
    examples = examples.reshape(-1, 28, 28)
    vis.vis_matrix(examples, s2x2 = True)
    print "----"

# Example Results:
#
# 0 Test Cost: 0.058131233367
#                             |                            |                            
#                             |                            |                            
#               ░░░░          |           ░░░░░░░░░        |            ░░░░░░░         
#           ░░░░░░░░░░░░      |        ░░░░▒▒▒▒▒▒░░░       |         ░░░░░▒▒░░░░░       
#         ░░░▒▒▒▒▒▒▒░░░░      |        ░░░░░░░▒▒▒░░░░      |        ░░░░░░▒▒▒▒░░░       
#        ░░░░░▒░░░░▒▒▒░       |       ░░░░░░░▒▒▒░░░░░      |       ░░░░░░░▒▒▒▒░░░       
#        ░░░░░░░░░▒▒▒░░       |        ░░░░░▒▒▒▒░░░░       |         ░░░░░▒▒▒░░░        
#        ░░░░░░▒▒▒▒▒░░        |        ░░░░▒▒▒▒░░░░░       |         ░░░▒▒▒▒▒░░░        
#        ░░░░░░░▒▓▒▒░░        |        ░░░░░░░░░░░░░░      |        ░░░░▒▒▒▒░░░░        
#          ░░░░▒▒▒▒░░         |       ░░░░░░░░░░░░░░░      |        ░░░░▒▒▒░░░░░        
#        ░░░░░░▒▒▒▒░          |       ░░░░░▒▒▒▒▒▒░░░       |       ░░░░░░▒▒▒░░░░        
#          ░░░░░░░░░          |        ░░▒▒▒▒▒▒░░░         |         ░░░▒░░░░░░       
#
# 
#   ...
#   ...
#
# 49 Test Cost: 0.0163615373844
#                             |                            |                            
#                             |            ░               |                            
#                             |       ░▄▓▓█▓▓▓▓▓░          |               ░▒▒░         
#        ░░░░░░░     ░        |      ░░▒▒░░  ░▒▒░          |               ░▒▒░         
#      ░▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒░      |              ▒▓▓░          |              ░▒▓░          
#       ░        ░░░▒▓▒       |            ░▒▓▒░           |              ░▓▒           
#                  ░▓▒░       |           ░▓▓▓▒░           |             ░▓▓░           
#            ░    ░▓▓░        |         ░▒██▓▒░            |            ░▓█▒            
#                ░▓▒░         |        ░▓██▓░░       ░░    |            ▒█▓░            
#               ▒▓▒           |       ░▓██▒░░░░   ░▄▓▒░    |           ░▓█▒             
#              ▓█▒            |       ░▓▓█████▓▓▓▓▓▓▓▒░    |           ▓█▓░             
#            ░▓█▓░            |         ░▀▀▀▀▀▀▀▀░░░       |          ░▓▓░              
#           ░▒▓▓░             |                            |                            
#           ░░░               |                            |                            


print ""
print "--------------------"
print ""

# Let's look at some filters.

W = net.layers[1].W.get_value()

for n in range(10):
    vis.vis_matrix_signed(W[:, n].reshape(28,28))

