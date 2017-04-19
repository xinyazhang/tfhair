#!/usr/bin/env python2

'''
Test for Contrained Hamilton's Principle
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
from copy import deepcopy
import math

def run():
    n = 3
    h = 1.0/1024.0
    learning_rate = 1e-4

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

    multiplier = tf.Variable(np.zeros(shape=[n], dtype=np.float32))
    g = TFGetLengthConstaintFunction(nrod)
    print('g shape: ', g.get_shape().as_list())
    gscalar = tf.reduce_sum(multiplier * g)
    #print('gscalar shape: ', gscalar.get_shape().as_list())

    L_i = tf.add(-TFGetEBend(crod), TFKineticI(prod, crod, h))
    L_ip1 = TFKineticI(crod, nrod, h) - TFGetEBend(nrod)
    print('L_i shape: ', L_i.shape.as_list())
    print('L_ip1 shape: ', L_ip1.shape.as_list())

    dL_i = tf.gradients(L_i, crod.xs)[0]
    dL_ip1 = tf.gradients(L_ip1, crod.xs)[0]
    dg = tf.gradients(g, nrod.xs)[0]
    print('dL_i shape: ', dL_i.shape.as_list())
    print('dL_ip1 shape: ', dL_ip1.shape.as_list())
    print('dg shape: ', dL_ip1.shape.as_list())
    dL = dL_i + dL_ip1 - multiplier * dg
    print('dL shape: ', dL.shape.as_list())
    loss1 = tf.reduce_sum(tf.multiply(dL, dL), axis=1)
    loss2 = g * g
    #loss2 = tf.reshape(g, [n])
    print('loss1 shape {}'.format(loss1.get_shape()))
    print('loss2 shape {}'.format(loss2.get_shape()))
    loss = tf.concat([loss1, loss2], axis=0)
    print('loss shape {}'.format(loss.get_shape()))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    thetas = np.zeros(shape=[n], dtype=np.float32)
    #'''
    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        [2,0,0],
        ])
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
    #'''
    rl = helper.calculate_rest_length(xs)
    #print(xsbar.shape)
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
        sess.run(nrod.xs.assign(xgt))
        print('--- Test with Initial value as GT ---')
        print('Initial value: ', sess.run(nrod.xs))
        print('Initial loss: ', sess.run(loss, feed_dict=inputdict))
        print('Initial L_i: ', sess.run(L_i, feed_dict=inputdict))
        print('Initial L_ip1: ', sess.run(L_ip1, feed_dict=inputdict))
        print('Initial dL_i: ', sess.run(dL_i, feed_dict=inputdict))
        print('Initial dL_ip1: ', sess.run(dL_ip1, feed_dict=inputdict))
        print('Initial g: ', sess.run(g, feed_dict=inputdict))
        print('--- Done ---')
        sess.run(nrod.xs.assign(xsbar))
        exiting_i = -1
        for i in range(10000):
            if i % 1000 == 0:
                print(sess.run(nrod.xs))
                print('loss: ', sess.run(loss, feed_dict=inputdict))
                # print('g: {}'.format(sess.run(g, feed_dict=inputdict)))
                print('multiplier: ', sess.run(multiplier, feed_dict=inputdict))
            sess.run(train_op, feed_dict=inputdict)
            loss_value = sess.run(tf.reduce_sum(loss), feed_dict=inputdict)
            if math.fabs(loss_value) < 1e-9:
                exiting_i = i
                break;
        converged = sess.run(nrod.xs)
        loss_value = sess.run(loss, feed_dict=inputdict)
        multiplier_value = sess.run(multiplier, feed_dict=inputdict)
        print('Converged after ', exiting_i, 'iterations at: ', converged)
        print('loss: ', converged)
        print('Ground truth: ', xgt)
        print('    Delta: ', xgt - converged)
        print('    Multipliers: ', multiplier_value)

    xs = np.array([
        [-1,-1,0],
        [0,0,0],
        [1,-1,0],
        [2,0,0],
        ])
    rl = helper.calculate_rest_length(xs)
    xsbar = np.array([
        [-1 + h,-1,0],
        [0 + h,0,0],
        [1 + h,-1,0],
        [2 + h,0,0],
        ])
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
        sess.run(nrod.xs.assign(xsbar))
        exiting_i = -1
        for i in range(10000):
            if i % 1000 == 0:
                print('xs: {}'.format(sess.run(nrod.xs)))
                print('loss: {}'.format(sess.run(loss, feed_dict=inputdict)))
                print('gscalar: {}'.format(sess.run(gscalar, feed_dict=inputdict)))
                print('multiplier: {}'.format(sess.run(multiplier, feed_dict=inputdict)))
            sess.run(train_op, feed_dict=inputdict)
            loss_value = sess.run(loss, feed_dict=inputdict)
            if math.fabs(loss_value) < 1e-9:
                exiting_i = i
                break;
        converged = sess.run(nrod.xs)
        loss_value = sess.run(loss, feed_dict=inputdict)
        multiplier_value = sess.run(multiplier, feed_dict=inputdict)
        print('Converged after ', exiting_i, 'iterations at: ', converged)
        print('loss: ', converged)
        print('Multipliers: ', multiplier_value)


if __name__ == '__main__':
    run()
