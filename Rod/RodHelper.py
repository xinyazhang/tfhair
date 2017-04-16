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
        xs_init = tf.Variable(xs_init([n+1, 3]))

    if thetas_init is None:
        thetas_init = tf.placeholder(tf.float32, shape=(n))
    else:
        thetas_init = tf.Variable(thetas_init([n]), shape=(n))

    restl = tf.placeholder(tf.float32, shape=[n])
    return ElasticRod.ElasticRod(xs_init, restl, thetas_init)

class RodSaver():

    def __init__(self, sess, feed_dict, save_as):
        self.sess = sess
        self.feed_dict = feed_dict
        self.dest = save_as

    def __enter__(self):
        self.rods = []
        return self

    def __exit__(self, type, value, traceback):
        # compute matrix size
        n_timesteps = len(self.rods)
        if n_timesteps == 0: return
        n_rods = len(self.rods[0])
        if n_rods == 0: return
        n_knots = self.rods[0][0].xs[0].get_shape()[0] + 1

        # save all rods into matrix
        results = np.zeros(shape=(n_timesteps, n_rods, n_knots, 4), dtype=np.float32)
        for i in xrange(n_timesteps):
            for j in xrange(n_rods):
                results[i,j,:  ,0:3] = self.sess.run(self.rods[i][j].xs, feed_dict=self.feed_dict)
                results[i,j,:-1,  3] = self.sess.run(self.rods[i][j].thetas, feed_dict=self.feed_dict)
        np.save(self.dest, results)

    def add_timestep(self, rods):
        self.rods.append(rods)
