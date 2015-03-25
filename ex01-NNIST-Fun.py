#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# Example 01 -- Playing with MNIST
#
# You may want to try this interactively in python.

# Data loading conveniences
import data
# Some visualization conveniences
import vis

# Import MNIST, the classic dataset for playing around with NNs.
# The first time you run this, it will download the dataset.
mnist = data.load("mnist")

# MNIST is split into 3 segments: train, valid, and test
# Each one consists of xs (images) and ys (labels)
print mnist

# Result:
#
# { 'test': { 'xs': <NumPy float32 array of shape (10000, 28, 28) >,
#             'ys': <NumPy int64 array of shape (10000,) > },
#   'train': { 'xs': <NumPy float32 array of shape (50000, 28, 28) >,
#              'ys': <NumPy int64 array of shape (50000,) > },
#   'valid': { 'xs': <NumPy float32 array of shape (10000, 28, 28) >,
#              'ys': <NumPy int64 array of shape (10000,) > } }

# Let's look at the xs and ys:

# Let's print the first to xs from the training set
# at half scale:
vis.vis_matrix(mnist["train"]["xs"][0:2], s2x2 = True)

# Result:
#
#                            |                            
#                            |                            
#                ░░▄ ▄▄▄░    |               ▄▓█▓▄        
#        ▄▄▄▓▓█████▒▒▀▀▀     |             ▄███▓▀█▄░      
#        ▀▓▀██▓▀▀▀▓          |           ▓██▓▓██░▒█▒      
#           ▒█▒              |         ░██▓▒  ░   ██░     
#            ▀█▄▄░           |        ▓█▀         ██▓     
#             ░▀▓█▓░         |       ▓█▒          ██▒     
#                ░██▓        |      ░█▓         ░▓█▒      
#             ░▄▓▓██▓        |      ░█▓      ░▄▓▓▀        
#          ▄▓███▓▀▀░         |      ░██▓▄▄▄▓█▓▒▀          
#     ▄▄▓████▓▀░             |       ▀▓███▓▀░             
#    ░▀▀▀▀░░                 |                            
#                            |                            

# And print the corresponding ys:
print mnist["train"]["ys"][0:2]

# Result:
#
# [5 0]
