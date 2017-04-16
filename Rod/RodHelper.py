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

def create_TFRod(n, xs_init=None, thetas_init=None):
    if xs_init is None:
        xs_init = tf.placeholder(tf.float32, shape=(n+1,3))
    else:
        xs_init = tf.Variable(xs_init)

    if thetas_init is None:
        thetas_init = tf.placeholder(tf.float32, shape=(n))
    else:
        thetas_init = tf.Variable(thetas_init, shape=(n))

    restl = tf.placeholder(tf.float32, shape=[n])
    return ElasticRod.ElasticRod(xs_init, restl, thetas_init)
