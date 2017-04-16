#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
from copy import deepcopy
import math

def Lagrangian(prod, crod, h):
    return TFKineticI(prod, crod, h) - TFGetEBend(crod) + TFGetCLength(prod, crod)

def discrete_EulerLagrangian(L1, L2, var):
    d2L = tf.gradients(L1, var)[0]
    d1L = tf.gradients(L2, var)[0]
    dL = d2L + d1L
    # dL_sq = tf.matmul(dL, tf.transpose(dL))
    # return tf.reduce_sum(dL_sq)
    return tf.reduce_sum(tf.multiply(dL, dL), axis=None, keep_dims=False)

def init_rods(n):
    rods = [ helper.create_TFRod(n) for i in range(n-1) ]
    rods.append(helper.create_TFRod(n, tf.zeros))
    return list(map(TFInitRod, rods))

def run():
    n = 3
    h = 1.0 / 1024.0
    learning_rate = 1e-3

    prod, crod, nrod = init_rods(n)

    xs = np.array([ [-1,0,0], [0,0,0], [1,0,0], [2,0,0] ])
    xh = np.array([ [ h,0,0], [h,0,0], [h,0,0], [h,0,0] ])
    xsbar = xs + xh
    xgt = xs + 2 * xh
    rl = helper.calculate_rest_length(xs)
    inputdict = {
        prod.xs     : xs,
        prod.thetas : np.zeros(shape=[n], dtype=np.float32),
        prod.restl  : rl,
        crod.xs     : xs + xh,
        crod.thetas : np.zeros(shape=[n], dtype=np.float32),
        crod.restl  : rl,
        # nrod.xs     : xs + 2 * xh,
        nrod.thetas : np.zeros(shape=[n], dtype=np.float32),
        nrod.restl  : rl,
    }

    L1 = Lagrangian(prod, crod, h)
    L2 = Lagrangian(crod, nrod, h)
    loss = discrete_EulerLagrangian(L1, L2, crod.xs)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(nrod.xs.assign((xsbar+xgt)/2))

        exiting_i = -1
        for i in xrange(10000000):
            # display
            if i % 1000 == 0:
                print 'loss:', sess.run(loss, feed_dict=inputdict)
            # optimize
            sess.run(train_op, feed_dict=inputdict)
            loss_value = sess.run(loss, feed_dict=inputdict)
            if math.fabs(loss_value) < 1e-9:
                exiting_i = i
                break;
        converged = sess.run(nrod.xs)
        loss_value = sess.run(loss, feed_dict=inputdict)
        print 'Converged after', exiting_i, 'iterations at:\n', converged
        print 'Ground truth\n', xgt
        print 'Delta: \n', xgt - converged
        print 'loss:', converged

        with helper.RodSaver(sess, inputdict, "rods_output.npy") as saver:
            saver.add_timestep([prod])    # add timestep i-1
            saver.add_timestep([crod])    # add timestep i
            saver.add_timestep([nrod])    # add timestep i+1
