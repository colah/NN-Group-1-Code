# -*- coding: utf-8 -*-

from theano import function, shared
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import softmax

#from NetworkUtil import *

charmap = {
    (1.0, 1.0,
     1.0, 1.0)  :  '█',
    (0.8, 0.8,
     0.8, 0.8)  :  '▓',
    (0.6, 0.6,
     0.6, 0.5)  :  '▒',
    (0.3, 0.3,
     0.3, 0.3)  :  '░',
    (0.0, 0.0,
     0.0, 0.0)  :  ' ',
    (1.0, 0.0,
     1.0, 1.0)  :  '▙',
    (1.0, 1.0,
     1.0, 0.0)  :  '▛',
    (1.0, 1.0,
     0.0, 1.0)  :  '▜',
    (0.0, 1.0,
     1.0, 1.0)  :  '▟',
    (0.0, 0.0,
     1.0, 1.0)  :  '▄',
    (1.0, 1.0,
     0.0, 0.0)  :  '▀',
    (0.0, 1.0,
     0.0, 1.0)  :  '▐',
    (1.0, 0.0,
     1.0, 0.0)  :  '▌',
    (0.0, 1.0,
     1.0, 0.0)  :  '▞',
    (1.0, 0.0,
     0.0, 1.0)  :  '▚',
    (0.0, 0.0,
     1.0, 0.0)  :  '▖',
    (0.0, 0.0,
     0.0, 1.0)  :  '▗',
    (1.0, 0.0,
     0.0, 0.0)  :  '▘',
    (0.0, 1.0,
     0.0, 0.0)  :  '▝'
    }

charmapH = {
    (1.0,
     1.0)  :  '█',
    (0.8,
     0.8)  :  '▓',
    (0.6,
     0.6)  :  '▒',
    (0.3,
     0.3)  :  '░',
    (0.0,
     0.0)  :  ' ',

    (0.0,
     1.0)  :  '▄',
    (1.0,
     0.0)  :  '▀',
    }

def get_char(v):
    if          v >  1.0: return '#'
    elif 1.0 >= v >  0.9: return '█'
    elif 0.9 >= v >  0.8: return '▓'
    elif 0.8 >= v >  0.5: return '▒'
    elif 0.5 >= v >  0.3: return '░'
    elif 0.3 >= v >  0.2: return ':'
    elif 0.2 >= v >  0.1: return '.'
    elif 0.1 >= v >= 0.0: return ' '
    else:                 return '?'

def get_char_2x2(a,b,c,d):
    C = '?'
    dist = 100
    #for a2,b2,c2,d2 in charmap2.keys():
    #    dist2 = (a-a2)**2 + (b-b2)**2 +\
    #            (c-c2)**2 + (d-d2)**2
    #    if dist2 < dist:
    #        C = charmap2[(a2,b2,c2,d2)]
    #        dist = dist2
    return get_char_2(a,c) + get_char_2(b,d)

def get_char_2(a,b):
    dist = 100
    C = "?"
    for a2,b2,in charmapH.keys():
        dist2 = (a-a2)**2 + (b-b2)**2
        if dist2 < dist:
            C = charmapH[a2,b2]
            dist = dist2
    return C


def vis_matrix(arrs, s2x2 = False):

    if len(arrs.shape) == 2:
        if arrs.shape[0] < 200 and arrs.shape[1] < 200:
            arrs = arrs.reshape(1, *arrs.shape)
        else:
            raise ValueError("vis_matrix_*: Shape of matrix to be visualized is too big. You probably made a mistake.")
    elif len(arrs.shape) == 1:
        raise ValueError("vis_matrix_* needs a 2 or 3 dimensional tensor. You probably want to use .reshape().")
    if arrs.shape[0] > 50:
        raise ValueError("vis_matrix_*: Can't visualize this many objects at once; you probably made a mistake.")

    N, Y, X = arrs.shape
    d = 2 if s2x2 else 1
    for y in range(0, Y, d):
        s = ""
        for n in range(N):
            for x in range(0, X, d):
                if s2x2:
                    if x == X-1 and y == Y-1:
                        s += get_char_2x2(arrs[n][y  ][x],   0,
                                          0,                 0)
                    elif x == X-11:
                        s += get_char_2x2(arrs[n][y  ][x],   0,
                                          arrs[n][y+1][x],   0)
                    elif y == Y-1:
                        s += get_char_2x2(arrs[n][y  ][x],   arrs[n][y  ][x+1],
                                          0,                 0)
                    else:
                        s += get_char_2x2(arrs[n][y  ][x],   arrs[n][y  ][x+1],
                                          arrs[n][y+1][x],   arrs[n][y+1][x+1])
                else:
                    s += 2*get_char(arrs[n][y][x])
            if n + 1 < N:
                s += "|"
        print s

def vis_matrix_signed(arrs):

    if len(arrs.shape) == 2:
        if arrs.shape[0] < 100 and arrs.shape[1] < 100:
            arrs = arrs.reshape(1, *arrs.shape)
        else:
            raise ValueError("vis_matrix_*: Shape of matrix to be visualized is too big. You probably made a mistake.")
    elif len(arrs.shape) == 1:
        raise ValueError("vis_matrix_* needs a 2 or 3 dimensional tensor. You probably want to use .reshape().")
    if arrs.shape[0] > 20:
        raise ValueError("vis_matrix_*: Can't visualize this many objects at once; you probably made a mistake.")

    N, Y, X = arrs.shape
    for y in range(Y):
        s = ""
        for n in range(N):
            for x in range(X):
                val = arrs[n][y][x]
                if val < 0:
                    s += "\033[91m"
                else:
                    s += "\033[34m"
                s += 2*get_char(abs(val))
            s += "\033[39m"
            if n + 1 < N:
                s += "|"
        print s
