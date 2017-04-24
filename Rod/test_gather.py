#!/usr/bin/env python2

import tensorflow as tf
import numpy as np

xs = tf.placeholder(shape=[5,3], dtype=tf.float32)
# sel = tf.placeholder(shape=[None, 1], dtype=tf.int32)
sel = tf.placeholder(shape=[None], dtype=tf.int32)
gat = tf.gather(xs, sel)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gat, feed_dict={xs:
       np.array([
           [0, -1, 7],
           [1, -2, 9],
           [2, -3, 11],
           [3, -4, 13],
           [4, -5, 17]
           ]),
       #sel : np.array([[3],[2],[4],[1],[0],[1],[2],[3],[4]])
       sel : np.array([3,2,4,1,0,1,2,3,4])
       }
       ))

tf.reset_default_graph()

xs2 = tf.placeholder(shape=[3,3,3], dtype=tf.float32)
sel2 = tf.placeholder(shape=[None, 2], dtype=tf.int32)
gat2 = tf.gather_nd(xs2, sel2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gat2, feed_dict={xs2:
       np.array([
           [[0, 0, 27],[0, 1, 11],[0, 2, 63]],
           [[1, 0, 13],[1, 1, 17],[1, 2, 51]],
           [[2, 0,  7],[2, 1, 31],[2, 2, 91]],
           ]),
       sel2 : np.array([[2,0],[2,1],[1,0],[1,1],[0,1],[0,0]])
       }
       ))
