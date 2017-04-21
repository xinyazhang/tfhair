#!/usr/bin/env python2
'''
Potential Energy
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi

def run_test1():
    '''
    Test 1: bending force only
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    irod = helper.create_TFRod(n)
    TFInitRod(irod)
    EBend = TFGetEBend(irod)
    Force = tf.gradients(-EBend, irod.xs)[0]

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [0,-1,0],
        [-1,-1,0],
        ])
    xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    rl = helper.calculate_rest_length(xs)
    thetas = np.zeros(shape=[n], dtype=np.float32)
    paddedthetas = np.zeros(shape=[n+1], dtype=np.float32)

    nxs = irod.xs + h * irod.xdots
    ndots = irod.xdots + h * Force #/ (rl * rho)
    saver = helper.RodSaver('/tmp/tftest1')
    for frame in range(720):
        with tf.Session() as sess:
            inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots}
            saver.add_timestep([xs], [paddedthetas])
            xs, xdots = sess.run([nxs, ndots], feed_dict=inputdict)
    saver.close()

def run_test2():
    '''
    Test 2: constant velocity
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    irod = helper.create_TFRod(n)
    TFInitRod(irod)
    E = TFGetEBend(irod)
    XForce = tf.gradients(-E, [irod.xs])[0]

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        [2,0,0],
        ])
    xdots = np.array([
        [1,10,0],
        [1,10,0],
        [1,10,0],
        [1,10,0],
        ])
    rl = helper.calculate_rest_length(xs)
    thetas = np.zeros(shape=[n], dtype=np.float32)
    paddedthetas = np.zeros(shape=[n+1], dtype=np.float32)

    nxs = irod.xs + h * irod.xdots
    ndots = irod.xdots + h * XForce #/ (rl * rho)
    saver = helper.RodSaver('/tmp/tftest2')
    for frame in range(720):
        with tf.Session() as sess:
            inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots}
            saver.add_timestep([xs], [paddedthetas])
            xs, xdots = sess.run([nxs, ndots], feed_dict=inputdict)
    saver.close()

def run():
    run_test1()
    run_test2()

if __name__ == '__main__':
    run()
