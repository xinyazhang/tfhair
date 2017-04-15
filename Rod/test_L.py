#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
from copy import deepcopy
import math

def run():
    n = 3
    h = 1.0/1024.0
    learning_rate = 1e-3

# q^{i-1}
    prod = helper.create_TFRod(n)
# q^i
    crod = helper.create_TFRod(n)
# q^{i+1}
    nrod = ElasticRod(tf.Variable(tf.zeros(crod.xs.shape.as_list()), name='q_ip1', dtype=tf.float32),
            tf.placeholder(tf.float32, shape=[n]),
            tf.placeholder(tf.float32, shape=[n]))
    #nrod.xs.set_shape(crod.xs.shape)
    nrod.xs.shape = crod.xs.shape
    TFInitRod(prod)
    TFInitRod(crod)
    TFInitRod(nrod)
    L_i = tf.add(-TFGetEBend(crod), TFKineticI(prod, crod, h))
    L_ip1 = tf.add(-TFGetEBend(nrod), TFKineticI(crod, nrod, h))
    print('L_i shape: ', L_i.shape.as_list())
    print('L_ip1 shape: ', L_ip1.shape.as_list())

    dL_i = tf.gradients(L_i, crod.xs)[0]
    dL_ip1 = tf.gradients(L_ip1, crod.xs)[0]
    dL = dL_i + dL_ip1
    print('dL_i shape: ', dL_i.shape.as_list())
    print('dL_ip1 shape: ', dL_ip1.shape.as_list())
    print('dL shape: ', dL.shape.as_list())
    loss = tf.reduce_sum(tf.multiply(dL, dL), axis=None, keep_dims=False)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # TODO: What is the best optimizer for our task?

    thetas = np.zeros(shape=[n], dtype=np.float32)
    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        [2,0,0],
        ])
    rl = helper.calculate_rest_length(xs)
    xsbar = np.array([
        [-1 + h,0,0],
        [0 + h,0,0],
        [1 + h,0,0],
        [2 + h,0,0],
        ])
    xgt = np.array([
        [-1 + 2 * h,0,0],
        [0 + 2 * h,0,0],
        [1 + 2 * h,0,0],
        [2 + 2 * h,0,0],
        ])
    print(xsbar.shape)
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

    print('Ground truth: ', xgt)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(nrod.xs.assign((xsbar+xgt)/2))
        '''
        print('Initial value: ', sess.run(nrod.xs))
        print('Initial loss: ', sess.run(loss, feed_dict=inputdict))
        print('Initial L_i: ', sess.run(L_i, feed_dict=inputdict))
        print('Initial L_ip1: ', sess.run(L_ip1, feed_dict=inputdict))
        print('Initial dL_i: ', sess.run(dL_i, feed_dict=inputdict))
        print('Initial dL_ip1: ', sess.run(dL_ip1, feed_dict=inputdict))
        exit()
        '''
        exiting_i = -1
        for i in range(10000):
            if i % 1000 == 0:
                print(sess.run(nrod.xs))
                print('loss: ', sess.run(loss, feed_dict=inputdict))
            sess.run(train_op, feed_dict=inputdict)
            loss_value = sess.run(loss, feed_dict=inputdict)
            if math.fabs(loss_value) < 1e-9:
                exiting_i = i
                break;
        converged = sess.run(nrod.xs)
        loss_value = sess.run(loss, feed_dict=inputdict)
        print('Converged after ', exiting_i, 'iterations at: ', converged)
        print('loss: ', converged)
        print('Ground truth: ', xgt)
        print('    Delta: ', xgt - converged)
