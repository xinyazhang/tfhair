#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
from ElasticRod import *
import RodHelper as helper

sess = tf.InteractiveSession()

testi=tf.placeholder(tf.float32, shape=[1])
testo=tf.gradients(-testi*testi, testi)[0]

print(testo.eval(feed_dict={testi:[0]}))

n = 3

# Input placeholder
irod = ElasticRod(tf.placeholder(tf.float32, shape=[n+1, 3]), tf.placeholder(tf.float32, shape=[n]), tf.placeholder(tf.float32, shape=[n]))

# Output tensor
EBend = TFGetEBend(irod)
ETwist = TFGetETwist(irod)
Force = tf.gradients(-EBend, irod.xs)
TForce = tf.gradients(-ETwist, irod.thetas)

sess.run(tf.global_variables_initializer())

xs=np.array([
    [-1,-1,0],
    [0,0,0],
    [1,-1,0],
    [2,0,0],
    ])
rl = helper.calculate_rest_length(xs)
print(rl)
thetas=np.zeros(shape=[n], dtype=np.float32)

inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas}
print(irod.ks.eval(feed_dict=inputdict))
print(EBend.eval(feed_dict=inputdict))
print('Twist %f' % ETwist.eval(feed_dict=inputdict))
print(Force[0].eval(feed_dict=inputdict))

xs2=np.array([
    [-1,-0.0,0],
    [0,0,0],
    [1,-0.0,0],
    [2,0.0,0],
    ])
rl2 = helper.calculate_rest_length(xs2)
inputdict = {irod.xs:xs2, irod.restl:rl2, irod.thetas:thetas}
print(EBend.eval(feed_dict=inputdict))
print(ETwist.eval(feed_dict=inputdict))
print(Force[0].eval(feed_dict=inputdict))

xs3=np.array([
    [-1,0.0,0],
    [0,0,0],
    [1,1e-5,0],
    [2,1e-5,0],
    ])
# print("Curvatures: ", irod.ks.eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
# print("Voronoi Length: ", irod.ls.eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))
# print("Energy w/o Voronoi Length: ", tf.norm(irod.ks, ord=2, axis=1).eval(feed_dict={irod.xs:xs3, irod.thetas:thetas}))

rl3 = helper.calculate_rest_length(xs3)
inputdict = {irod.xs:xs3, irod.restl:rl3, irod.thetas:thetas}
print(EBend.eval(feed_dict=inputdict))
print(ETwist.eval(feed_dict=inputdict))
print(Force[0].eval(feed_dict=inputdict))

xs4=np.array([
    [-1,0.0,0],
    [0,0,0],
    [1,-1e-5,0],
    [2,-1e-5,0],
    ])
rl4 = helper.calculate_rest_length(xs4)
inputdict = {irod.xs:xs4, irod.restl:rl4, irod.thetas:thetas}
print(EBend.eval(feed_dict=inputdict))
print(ETwist.eval(feed_dict=inputdict))
print(Force[0].eval(feed_dict=inputdict))

xs5=np.array([
    [-1,0.0,0],
    [0,0,0],
    [1,0,0],
    [2,1,0],
    ])
rl5 = helper.calculate_rest_length(xs5)
inputdict = {irod.xs:xs5, irod.restl:rl5, irod.thetas:thetas}
print(EBend.eval(feed_dict=inputdict))
print(ETwist.eval(feed_dict=inputdict))
print(Force[0].eval(feed_dict=inputdict))
