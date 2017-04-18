#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf

def Lagrangian(prod, crod, h):
    return TFKineticI(prod, crod, h) - TFGetEBend(crod) - prod.multiplier * TFGetCLength(prod)

def discrete_EulerLagrangian(L1, L2, var):
    d2L = reduce(tf.add, filter(lambda x: x is not None, tf.gradients(L1, var)))
    d1L = reduce(tf.add, filter(lambda x: x is not None, tf.gradients(L2, var)))
    dL = d2L + d1L
    return tf.reduce_sum(tf.multiply(dL, dL), axis=1, keep_dims=False)

def run():
    # global settings
    n = 3
    h = 1.0 / 1024.0
    learning_rate = 1e-5

    # initial values
    thetas = np.zeros(shape=[n], dtype=np.float32)
    xs = np.array([ [-1,0,0], [0,0,0], [1,0,0], [2,0,0] ])
    xh = np.array([ [ h,0,0], [h,0,0], [h,0,0], [h,0,0] ])
    #perturb = np.array([ [ h, 0,0], [0,0,0], [0,0,0], [0,0,0] ])
    # TODO: zero perturb should return zero loss instantly, but now it isn't
    perturb = np.array([ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ])
    xsbar = xs + xh
    xinit = xsbar + perturb
    rl = helper.calculate_rest_length(xs)

    # instantiate rods of different timestep
    prod = helper.create_TFRod_constant(n, xs, thetas, rl)
    prod.multiplier = tf.zeros(xs.shape, dtype=tf.float32)
    crod = helper.create_TFRod_constant(n, xsbar, thetas, rl)
    crod.multiplier = tf.Variable(tf.zeros(xs.shape), dtype=tf.float32)
    nrod = helper.create_TFRod_variable(n, tf.zeros, tf.zeros, rl)
    prod, crod, nrod = map(TFInitRod, [prod, crod, nrod])

    # optimizer
    L1 = Lagrangian(crod, nrod, h)
    L2 = Lagrangian(prod, crod, h)
    dL1 = tf.gradients(L1, [crod.xs, crod.multiplier])
    print(dL1)
    dL2 = tf.gradients(L2, crod.xs)
    print(dL2)
    loss_vector = dL1[0] + dL1[1] + dL2[0]
    loss = tf.reduce_sum(tf.multiply(loss_vector, loss_vector), axis=1, keep_dims=False)
    print('loss shape: {}'.format(loss.get_shape()))
    print('Newton method: {}'.format(tf.gradients(loss, [crod.xs, crod.multiplier])))
    #loss = discrete_EulerLagrangian(L1, L2, [crod.xs, crod.multiplier])
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(nrod.xs.assign(xinit))

        with helper.Trainer(sess, train_op, loss) as t:
            converged = sess.run(nrod.xs)
            new_rl = helper.calculate_rest_length(converged)
            print 'Expected: {}'.format(rl)
            print 'Actual  : {}'.format(new_rl)
            print 'Delta: {}'.format(rl - new_rl)

        with helper.RodSaver(sess, "rods_output.npy") as saver:
            saver.add_timestep([prod])    # add timestep i-1
            saver.add_timestep([crod])    # add timestep i
            saver.add_timestep([nrod])    # add timestep i+1
