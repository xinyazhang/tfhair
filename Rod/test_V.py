#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi

def run_bend():
    n = 3
# Input placeholder
    irod = helper.create_TFRod(n)
# Output tensor
    TFInitRod(irod)
    EBend = TFGetEBend(irod)
    Force = tf.gradients(-EBend, irod.xs)

    xs=np.array([
        [-1,-1,0],
        [0,0,0],
        [1,-1,0],
        [2,0,0],
        ])
    rl = helper.calculate_rest_length(xs)
    print(rl)
    thetas=np.zeros(shape=[n], dtype=np.float32)

    xs2=np.array([
        [-1,-0.0,0],
        [0,0,0],
        [1,-0.0,0],
        [2,0.0,0],
        ])
    rl2 = helper.calculate_rest_length(xs2)

    xs3=np.array([
        [-1,0.0,0],
        [0,0,0],
        [1,1e-5,0],
        [2,1e-5,0],
        ])
    rl3 = helper.calculate_rest_length(xs3)

    xs4=np.array([
        [-1,0.0,0],
        [0,0,0],
        [1,-1e-5,0],
        [2,-1e-5,0],
        ])
    rl4 = helper.calculate_rest_length(xs4)

    xs5=np.array([
        [-1,0.0,0],
        [0,0,0],
        [1,0,0],
        [2,1,0],
        ])
    rl5 = helper.calculate_rest_length(xs5)
    with tf.Session() as sess:
        inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas}
        print(irod.ks.eval(feed_dict=inputdict))
        print('Bend %f Expecting: %f' % (EBend.eval(feed_dict=inputdict) , 4 * math.sqrt(2)) )
        print(Force[0].eval(feed_dict=inputdict))

        inputdict = {irod.xs:xs2, irod.restl:rl2, irod.thetas:thetas}
        print('Bend %f Expecting: %f' % (EBend.eval(feed_dict=inputdict) , 0) )
        print(Force[0].eval(feed_dict=inputdict))

        inputdict = {irod.xs:xs3, irod.restl:rl3, irod.thetas:thetas}
        print(EBend.eval(feed_dict=inputdict))
        print(Force[0].eval(feed_dict=inputdict))

        inputdict = {irod.xs:xs4, irod.restl:rl4, irod.thetas:thetas}
        print(EBend.eval(feed_dict=inputdict))
        print(Force[0].eval(feed_dict=inputdict))

        inputdict = {irod.xs:xs5, irod.restl:rl5, irod.thetas:thetas}
        print(EBend.eval(feed_dict=inputdict))
        print(Force[0].eval(feed_dict=inputdict))


def run_twist():
    n = 2
    rod = helper.create_TFRod(n)
    rod.refd1s = tf.placeholder(tf.float32, shape=[n-1,3])
    rod.refd2s = tf.placeholder(tf.float32, shape=[n-1,3])
    TFInitRod(rod)
    ETwist = TFGetETwist(rod)
    XForce = tf.gradients(-ETwist, rod.xs)[0]
    TForce = tf.gradients(-ETwist, rod.thetas)[0]

    xs = np.array([
        [0,0.0,0],
        [1,0,0],
        [1,-1,0]
        ])
    rl = helper.calculate_rest_length(xs)
    refd1s, refd2s = helper.calculate_referene_directions(xs, np.array([0,1,0]))
    '''
    refd1s = np.array([
        [0,1,0],
    #    [1,0,0],
        ])
    refd2s = np.array([
        [0,0,1],
    #    [0,0,1],
        ])
    '''
    print('refd1s {}'.format(refd1s))
    print('refd2s {}'.format(refd2s))
    thetas = np.array([0, 2 * pi])

    with tf.Session() as sess:
        tf.global_variables_initializer()
        inputdict = { rod.xs : xs,
                rod.restl : rl,
                rod.thetas : thetas,
                rod.refd1s : refd1s[:-1],
                rod.refd2s : refd2s[:-1]}
        print('Twist %f' % ETwist.eval(feed_dict=inputdict))
        print('Force on thetas {}'.format(TForce.eval(feed_dict=inputdict)))
        print('Force on xs {}'.format(XForce.eval(feed_dict=inputdict)))

def run():
    run_bend()
    run_twist()

if __name__ == '__main__':
    run()
