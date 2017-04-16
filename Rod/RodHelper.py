#!/usr/bin/env python2

import numpy as np
import ElasticRod
import tensorflow as tf
import math

def calculate_rest_length(xs):
    nvert = xs.shape[0]
    e = xs[1:,:] - xs[0:-1,:]
    return np.sum(np.abs(e)**2, axis=-1)**(1./2)

def create_TFRod(n):
    return ElasticRod.ElasticRod(tf.placeholder(tf.float32, shape=[n+1, 3]),
            tf.placeholder(tf.float32, shape=[n]),
            tf.placeholder(tf.float32, shape=[n]))

def create_TFRod_variable(n, xs_init, thetas_init, restl):
    xs_init = tf.Variable(xs_init([n+1, 3]))
    thetas_init = tf.Variable(thetas_init([n]))
    restl = tf.constant(restl, dtype=tf.float32)
    return ElasticRod.ElasticRod(xs_init, restl, thetas_init)

def create_TFRod_constant(n, xs_init, thetas_init, restl):
    xs_init = tf.constant(xs_init, dtype=tf.float32)
    thetas_init = tf.constant(thetas_init, dtype=tf.float32)
    restl = tf.constant(restl, dtype=tf.float32)
    return ElasticRod.ElasticRod(xs_init, restl, thetas_init)

class RodSaver():

    def __init__(self, sess, save_as, feed_dict=None):
        self.sess = sess
        self.dest = save_as
        self.feed_dict = feed_dict

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

class Trainer():

    def __init__(self, sess, opt, loss, feed_dict=None,
            max_iter=100000, epsilon=1e-9, display=1000):
        self.sess = sess
        self.opt = opt
        self.loss = loss
        self.feed_dict = feed_dict
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.display = 1000

    def __enter__(self):
        self.curr_iter = self.max_iter
        for i in xrange(self.max_iter):
            # display loss over iterations
            if i % self.display == 0:
                print "--> loss at epoch %d:" % i, self.sess.run(self.loss, feed_dict=self.feed_dict)
            # optimize loss metrics
            self.sess.run(self.opt, feed_dict=self.feed_dict)
            loss_value = self.sess.run(self.loss, feed_dict=self.feed_dict)
            if math.fabs(loss_value) < self.epsilon:
                self.curr_iter = i
                print "--> loss at epoch %d:" % i, self.sess.run(self.loss, feed_dict=self.feed_dict)
                return self
        return self

    def __exit__(self, type, value, traceback):
        pass
