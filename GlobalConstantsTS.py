# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:04:25 2019

@author: Piotr Z. Jelonek
"""
from math import ceil, log

# defnining global parameters of the simulation
class gc_ts():
    # definition of a 'small' number
    small=10**(-10)
    # definition of 'large' number
    large=1000000
    # maximal number of rejections in BM
    maxiter=5000
    # maximal order of a quadrature in Devroye
    maxorder=500
    # set to 1 if standard deviation is supposed to be one
    normalised=True
    # output (on/off)
    outpt=True
    # number of points for a pdf (FFT works best for powers of 2)
    N=4000; N=2**ceil(log(N,2))