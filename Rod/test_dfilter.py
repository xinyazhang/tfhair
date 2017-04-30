#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import ElasticRod as ER

h = 1.0/1024.0
xs = tf.placeholder(shape=[2,3,3], dtype=tf.float32)
sqnxs = ER._dot(xs, xs, keep_dims=True)
thresh = tf.constant(2.1/h)
tensors = ER.TFSegmentVsSegmentDistanceFilter(h, 0, 2, xs, sqnxs, thresh, None)

xsd = np.array([
    [[-1, 0, 0],[0,0,0],[1,0,0]],
    [[-1, -1, 1],[-1,0,1],[-1,1,1]]
    ])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run(tensors, feed_dict={xs:xsd})
    for r in results:
        print(r)
