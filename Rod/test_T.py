#!/usr/bin/env python2
'''
Kinetic Energy
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import math

def run():
    n = 3
    h = 1.0/1024.0

    rodthis = helper.create_TFRod(n)
    rodnext = helper.create_TFRod(n)
    TFInitRod(rodthis)
    TFInitRod(rodnext)
    kinetic = TFKineticI(rodthis, rodnext, h)
    rodd = helper.create_TFRod(n)

    rodd.xdots = tf.placeholder(tf.float32, shape=[n+1,3])
    kineticD = TFKineticD(rodd)

    thetas = np.zeros(shape=[n], dtype=np.float32)
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
    xdots = np.array([
        [0,-1,0],
        [0,-1,0],
        [0,-1,0],
        [0,-1,0]
        ])
    with tf.Session() as sess:
        tf.global_variables_initializer()
        inputdict = {
                rodthis.xs : xs,
                rodthis.restl : rl,
                rodthis.thetas : thetas,
                rodnext.xs : xsbar,
                rodnext.restl : rl,
                rodnext.thetas : thetas
                }
        print('Kinetic energy %f Expecting: %f' %
                (kinetic.eval(feed_dict=inputdict), 0.5 * 3 * math.sqrt(2))
             )
        inputdict = {
                rodd.xs : xs,
                rodd.restl : rl,
                rodd.thetas : thetas,
                rodd.xdots : xdots
                }
        print('Kinetic energy %f Expecting: %f' %
               (kineticD.eval(feed_dict=inputdict), 0.5 * 3 * math.sqrt(2)))
    # print(rodthis.fullrestvl.eval(feed_dict=inputdict))
    # print(rodthis.xdots.eval(feed_dict=inputdict))

if __name__ == '__main__':
    run()
