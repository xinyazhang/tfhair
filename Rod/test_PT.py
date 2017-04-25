#!/usr/bin/env python2
'''
Propogation of Twisting
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi

def run():
    n = 3
    rod = helper.create_TFRod(n)
    rod.refd1s = tf.placeholder(tf.float32, shape=[n,3])
    rod.refd2s = tf.placeholder(tf.float32, shape=[n,3])
    nrod = helper.create_TFRod(n)
    TFInitRod(rod)
    TFInitRod(nrod)
    TFPropogateRefDs(rod, nrod)

    xs = np.array([
        [0,0.0,0],
        [1,0,0],
        [1,-1,0],
        [1,-2,0]
        ])
    nxs = np.array([
        [0,0.0,0],
        [0.866025403784438647, 0.500000000000000000,0],
        [1.36602540378443865, -0.366025403784438647,0],
        [1.86602540378443865, -1.23205080756887729, 0]
        ])
    rl = helper.calculate_rest_length(xs)
    refd1s, refd2s = helper.calculate_reference_directions(xs, np.array([0,1,0]))
    expnrefd1s, expnrefd2s = helper.calculate_reference_directions(nxs, np.array([-0.500000000000000000, 0.866025403784438647,0]))
    # print('refd1s {}'.format(refd1s))
    # print('refd2s {}'.format(refd2s))
    thetas = np.array([0, 0, 0])

    with tf.Session() as sess:
        tf.global_variables_initializer()
        inputdict = { rod.xs : xs,
                rod.restl : rl,
                rod.thetas : thetas,
                rod.refd1s : refd1s,
                rod.refd2s : refd2s,
                nrod.xs : nxs,
                nrod.restl : rl,
                nrod.thetas : thetas
                }
        nrefd1s = nrod.refd1s.eval(feed_dict=inputdict)
        nrefd2s = nrod.refd2s.eval(feed_dict=inputdict)
        print("next refd1s {}\nExp {}\nDelta {}".format(nrefd1s, expnrefd1s, nrefd1s - expnrefd1s))
        print("next refd2s {}\nExp {}\nDelta {}".format(nrefd2s, expnrefd2s, nrefd2s - expnrefd2s))

if __name__ == '__main__':
    run()
