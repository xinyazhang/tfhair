#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf

# def Lagrangian(prod, crod, h, multiplier=None):
#     if multiplier is None:
#         return TFKineticI(prod, crod, h) - TFGetEBend(crod)
#     else:
#         return TFKineticI(prod, crod, h) - TFGetEBend(crod) - multiplier * TFGetCLength(crod)

def Lagrangian(prod, crod, h):
    return TFKineticI(prod, crod, h) - TFGetEBend(crod) + TFGetCLength(crod)

def discrete_EulerLagrangian(L1, L2, var):
    d2L = reduce(tf.add, filter(lambda x: x is not None, tf.gradients(L1, var)))
    d1L = reduce(tf.add, filter(lambda x: x is not None, tf.gradients(L2, var)))
    dL = d2L + d1L
    return tf.reduce_sum(dL * dL)

def run():
    # global settings
    n = 3
    h = 1.0 / 1024.0
    learning_rate = 1e-3

    # initial values
    thetas = np.zeros(shape=[n], dtype=np.float32)
    xs = np.array([ [-1,0,0], [0,0,0], [1,0,0], [2,0,0] ])
    xh = np.array([ [ h,0,0], [h,0,0], [h,0,0], [h,0,0] ])
    perturb = np.array([ [h,0,0], [0,0,0], [0,0,0], [0,0,0] ])
    # perturb = np.array([ [0,0,0], [h,0,0], [0,0,0], [0,0,0] ])
    # perturb = np.array([ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ])
    xsbar = xs + xh
    xinit = xs + 2 * xh + perturb
    rl = helper.calculate_rest_length(xs)

    # instantiate rods of different timestep
    prod = helper.create_TFRod(n)
    crod = helper.create_TFRod(n)
    nrod = helper.create_TFRod_variable(n, tf.zeros, tf.zeros, rl)
    prod, crod, nrod = map(TFInitRod, [prod, crod, nrod])

    multiplier = tf.Variable(np.zeros(shape=[n], dtype=np.float32))

    # optimizer
    L1 = Lagrangian(prod, crod, h)
    L2 = Lagrangian(crod, nrod, h)
    loss = discrete_EulerLagrangian(L1, L2, crod.xs)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    inputdict = {
        prod.xs : xs,
        prod.restl : rl,
        prod.thetas : thetas,
        crod.xs : xsbar,
        crod.restl : rl,
        crod.thetas : thetas,
        nrod.restl : rl,
        nrod.thetas : thetas,
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(nrod.xs.assign(xinit))

        with helper.Trainer(sess, train_op, loss, inputdict) as t:
            converged = sess.run(nrod.xs)
            new_rl = helper.calculate_rest_length(converged)
            print 'Multipliers: {}'.format(sess.run(multiplier))
            print 'Expected: {}'.format(rl)
            print 'Actual  : {}'.format(new_rl)
            print 'Delta: {}'.format(rl - new_rl)

        with helper.RodSaver(sess, "rods_output.npy", inputdict) as saver:
            saver.add_timestep([prod])    # add timestep i-1
            saver.add_timestep([crod])    # add timestep i
            saver.add_timestep([nrod])    # add timestep i+1
