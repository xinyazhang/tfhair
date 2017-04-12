#!/usr/bin/env python2

import numpy as np

def calculate_rest_length(xs):
    nvert = xs.shape[0]
    e = xs[1:,:] - xs[0:-1,:]
    return np.sum(np.abs(e)**2, axis=-1)**(1./2)
