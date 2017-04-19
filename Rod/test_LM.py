#!/usr/bin/env python2

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf

x = tf.Variable(tf.ones([1]), dtype=tf.float32)
y = tf.Variable(tf.ones([1]), dtype=tf.float32)
l = tf.Variable(tf.ones([1]), dtype=tf.float32)

f1 = 2 * x * y
f2 = x*x + y*y - 50
L = f1 - l * f2
dL = reduce(tf.add, tf.gradients(L, [x,y,l]))
L2 = dL * dL

opt = tf.train.AdamOptimizer(1e-3).minimize(L2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(L2)
    while sess.run(L2) > 1e-9:
        sess.run(opt)
        value = sess.run(L2)
        print value
    print sess.run([x,y,l])
    print sess.run([L2])
    print sess.run([f2]), "== 0?"
