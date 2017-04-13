#!/usr/bin/env python2

import numpy as np
import ElasticRod
import tensorflow as tf

def calculate_rest_length(xs):
    nvert = xs.shape[0]
    e = xs[1:,:] - xs[0:-1,:]
    return np.sum(np.abs(e)**2, axis=-1)**(1./2)

def create_TFRod(n):
    return ElasticRod.ElasticRod(tf.placeholder(tf.float32, shape=[n+1, 3]),
            tf.placeholder(tf.float32, shape=[n]),
            tf.placeholder(tf.float32, shape=[n]))
