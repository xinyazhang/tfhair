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
Force = tf.gradients(-tf.norm(EBend, ord=2), irod.xs)

sess.run(tf.global_variables_initializer())

xs=np.array([
    [-1,-0.1,0],
    [0,0,0],
    [1,-0.1,0],
    [2,0.1,0],
    ])
thetas=np.empty(shape=[n], dtype=np.float32)

print(EBend.eval(feed_dict={irod.xs:xs, irod.thetas:thetas}))
print(Force[0].eval(feed_dict={irod.xs:xs, irod.thetas:thetas}))

xs2=np.array([
    [-1,-0.0,0],
    [0,0,0],
    [1,-0.0,0],
    [2,0.0,0],
    ])
print(EBend.eval(feed_dict={irod.xs:xs2, irod.thetas:thetas}))
print(Force[0].eval(feed_dict={irod.xs:xs2, irod.thetas:thetas}))

xs3=np.array([
    [-1,0.0,0],
    [0,0,0],
    [1,0.0,0],
    [2,1e-15,0],
    ])
# print("Curvatures: ", irod.ks.eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
# print("Voronoi Length: ", irod.ls.eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
# print("Energy w/o Voronoi Length: ", tf.norm(irod.ks, ord=2, axis=1).eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))

print(EBend.eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
print(Force[0].eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
