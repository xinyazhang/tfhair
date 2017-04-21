#!/usr/bin/env python2

import os
import shutil
import numpy as np
import ElasticRod
import tensorflow as tf
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def calculate_norms(e):
    return np.sum(np.abs(e)**2, axis=-1)**(1./2)

def divide_vecarray_by_scalararray(numerator, denominator):
    return np.divide(numerator.T, denominator).T

def normalize(e):
    return divide_vecarray_by_scalararray(e, calculate_norms(e))

def calculate_rest_length(xs):
    #nvert = xs.shape[0]
    e = xs[1:,:] - xs[0:-1,:]
    return calculate_norms(e)

def calculate_kb(e):
    norms = np.sum(np.abs(e)**2, axis=-1)**(1./2)
    e_i_1 = e[0:-1,:]
    e_i = e[1:,:]
    kb_numerator = 2 * np.cross(e_i_1, e_i)
    kb_denominator = np.multiply(norms[:-1], norms[1:]) + np.einsum('ij,ij->i', e_i_1, e_i)
    kb_denominator.reshape([kb_denominator.shape[0],1])
    # print('kb_numerator {}'.format(kb_numerator))
    # print('kb_denominator {}'.format(kb_denominator))
    return divide_vecarray_by_scalararray(kb_numerator, kb_denominator)

def calculate_parallel_transport(eprev, ethis):
    vector = np.cross(eprev, ethis)
    norm = math.fabs(calculate_norms(vector))
    if norm < 1e-9:
        return np.identity(3)
    axis = vector
    cosine = np.dot(eprev, ethis)
    theta = math.acos(cosine)
    return rotation_matrix(axis, theta)

def calculate_referene_directions(xs, initd1):
    e = xs[1:,:] - xs[0:-1,:]
    ebar = normalize(e)
    kb = calculate_kb(e)
    # print('edge {}'.format(e))
    # print('kb {}'.format(kb))
    prevd1 = initd1
    prevd2 = np.cross(e[0], initd1)
    # print('initd1 {}'.format(prevd1))
    # print('initd2 {}'.format(prevd2))
    d1arr = [prevd1]
    d2arr = [prevd2]
    for i in range(1, e.shape[0]):
        P = calculate_parallel_transport(ebar[i-1], ebar[i])
        d1 = P.dot(prevd1)
        d2 = P.dot(prevd2)
        # print('d1[{}]: {}'.format(i, d1))
        # print('d2[{}]: {}'.format(i, d2))
        d1arr.append(normalize(d1))
        d2arr.append(normalize(d2))
        prevd1 = d1
        prevd2 = d2
    return np.array(d1arr), np.array(d2arr)

def create_TFRod(n):
    rod = ElasticRod.ElasticRod(tf.placeholder(tf.float32, shape=[n+1, 3]),
            tf.placeholder(tf.float32, shape=[n]),
            tf.placeholder(tf.float32, shape=[n]))
    rod.xdots = tf.placeholder(tf.float32, shape=[n+1, 3])
    return rod

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

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.frame = 0
        # clean dest directory
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass

    def add_timestep(self, cpos, theta):
        # save all rods into matrix
        n_rods = len(cpos)
        n_knots = cpos[0].shape[0]

        results = np.zeros(shape=(n_rods, n_knots, 4), dtype=np.float32)
        for j in xrange(n_rods):
            results[j,:,0:3] = cpos[j]
            results[j,:,  3] = theta[j] #np.reshape(theta[j], (4))
        filename = os.path.join(self.directory, "%d.npy" % self.frame)
        np.save(filename, results)
        self.frame += 1

class Trainer():

    def __init__(self, sess, opt, loss, feed_dict=None,
            max_iter=100000, epsilon=1e-9, display=1000):
        self.sess = sess
        self.opt = opt
        self.loss = loss
        self.scalar_loss = tf.reduce_sum(loss, axis=None, keep_dims=False)
        print('scalar_loss shape {}'.format(self.scalar_loss.get_shape()))
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
            loss_value = self.sess.run(self.scalar_loss, feed_dict=self.feed_dict)
            if math.fabs(loss_value) < self.epsilon:
                self.curr_iter = i
                print("--> loss at epoch {}: {}".format(i, loss_value))
                return self
        return self

    def __exit__(self, type, value, traceback):
        pass
