#!/usr/bin/env python2

import tensorflow as tf
import numpy as np

xs = tf.placeholder(shape=[None,3], dtype=tf.float32)
# sel = tf.placeholder(shape=[None, 1], dtype=tf.int32)
rs = tf.reduce_sum(xs, axis=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(rs, feed_dict={xs:
       np.array([
           [0, -1, 7],
           [1, -2, 9],
           [2, -3, 11],
           [3, -4, 13],
           [4, -5, 17]
           ])
       }
       ))

tf.reset_default_graph()
