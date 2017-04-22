#!/usr/bin/env python2

import sys
import math
import numpy as np
import tensorflow as tf
import RodHelper as helper

from ElasticRod import *

# global settings
n = 5               # number of segments
h = 1.0/30.0
alpha = 1e-3            # learning rate

# rods objects
crod = ElasticRod(
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.cpos"),
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.cvel"),
    tf.placeholder(shape=(n+1, 1), dtype=tf.float32, name="crod.theta"),
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.omega"),
    tf.placeholder(shape=(n),      dtype=tf.float32, name="crod.restl"))
nrod = crod.next_state(h)

# constraints
# crod.init_length_constraint()

# initial values
cpos_value = np.asarray([ [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0], [6,0,0] ])
cvel_value = np.asarray([ [0,1,0], [0,1,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0] ])

# inputs to tensforlow
feeds = {
    crod.cpos : cpos_value,
    crod.cvel : cvel_value,
    crod.theta: np.zeros([n+1,1]),
    crod.omega: np.zeros([n+1,3]),
    crod.restl: np.ones(n),
}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = helper.RodSaver("/tmp/tfhair/")
    for i in range(100):
        print >> sys.stderr, "iteration i = %d" % i
        crod.dump(sess, feeds, name="crod")
        new_feeds = {
            crod.cpos : sess.run(nrod.cpos, feed_dict=feeds),
            crod.cvel : sess.run(nrod.cvel, feed_dict=feeds),
            crod.theta: sess.run(nrod.theta, feed_dict=feeds),
            crod.omega: sess.run(nrod.omega, feed_dict=feeds),
            crod.restl: feeds[crod.restl]
        }
        feeds = new_feeds
        saver.add_timestep(
            [ new_feeds[crod.cpos] ],
            [ new_feeds[crod.theta ] ])
    saver.close()
