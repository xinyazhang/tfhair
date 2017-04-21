#!/usr/bin/env python2

import math
import numpy as np
import tensorflow as tf
import RodHelper as helper

from ElasticRod import *
from functools import partial

# global settings
n = 3               # number of segments
# h = 1.0/1024.0      # timestep
h = 1.0/30.0
alpha = 1e-3            # learning rate

# rods objects
crod = ElasticRod(
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.cpos"),
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.cvel"),
    tf.placeholder(shape=(n+1, 1), dtype=tf.float32, name="crod.theta"),
    tf.placeholder(shape=(n+1, 3), dtype=tf.float32, name="crod.omega"))

nrod = ElasticRod(
    crod.update_cpos(h),
    crod.update_cvel(h),
    crod.update_theta(h),
    crod.update_omega(h))

# initial values
# cpos_value = np.asarray([ [1,0,0], [2,0,0], [3,0,0], [4,0,0] ])
# cpos_value = np.asarray([ [1,0,0], [2,0,0], [4,0,0], [7,0,0] ])
cpos_value = np.asarray([ [1,1,0], [2,-1,0], [3,1,0], [4,-1,0] ])
# cvel_value = np.asarray([ [1,0,0], [1,0,0], [1,0,0], [1,0,0] ])
cvel_value = np.asarray([ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ])

# inputs to tensforlow
feeds = {
    crod.cpos : cpos_value,
    crod.cvel : cvel_value,
    crod.theta: np.zeros([n+1,1]),
    crod.omega: np.zeros([n+1,3]),
}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rod_init = partial(ElasticRod.init, sess=sess, feeds=feeds)
    crod, nrod = map(rod_init, [crod, nrod])
    saver = helper.RodSaver("/tmp/tfhair/")
    for i in range(1000):
        # crod.dump(sess, feeds, name="nrod")
        new_feeds = {
            crod.cpos : sess.run(nrod.cpos, feed_dict=feeds),
            crod.cvel : sess.run(nrod.cvel, feed_dict=feeds),
            crod.theta: sess.run(nrod.theta, feed_dict=feeds),
            crod.omega: sess.run(nrod.omega, feed_dict=feeds)
        }
        feeds = new_feeds
        saver.add_timestep(
            [ new_feeds[crod.cpos] ],
            [ new_feeds[crod.theta ] ])
    saver.close()
