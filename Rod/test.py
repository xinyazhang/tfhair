#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
from ElasticRod import *

sess = tf.InteractiveSession()

n = 3

# Input placeholder
irod = ElasticRod(tf.placeholder(tf.float32, shape=[n+1, 3]), tf.placeholder(tf.float32, shape=[n]))

# Output tensor
EBend = TFGetEBend(irod)

sess.run(tf.global_variables_initializer())

xs=np.array([
    [-1,-0.1,0],
    [0,0,0],
    [1,-0.1,0],
    [2,0.1,0],
    ])
thetas=np.empty(shape=[n], dtype=np.float32)

print(EBend.eval(feed_dict={irod.xs:xs, irod.thetas:thetas}))

xs2=np.array([
    [-1,-0.0,0],
    [0,0,0],
    [1,-0.0,0],
    [2,0.0,0],
    ])
print(EBend.eval(feed_dict={irod.xs:xs2, irod.thetas:thetas}))
