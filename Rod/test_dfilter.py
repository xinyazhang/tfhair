#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
from numpy import linalg as la
import ElasticRod as ER

def print_exp(xs, thresh, chosen_seg):
    dims = xs.shape
    seglist = []
    for i0 in range(dims[0]):
        for j0 in range(dims[1]):
            seglist.append([i0, j0])
    nseg = len(seglist)
    for s0 in range(nseg):
        for s1 in range(s0+1, nseg):
            [i0, j0] = seglist[s0]
            [i1, j1] = seglist[s1]
            if chosen_seg is not None and j1 != chosen_seg:
                continue
            n = la.norm(xs[i0,j0] - xs[i1,j1])
            if n < thresh:
                print('{} distance: {}'.format([i0,j0,i1,j1], n))

# print(np.int64(1))
h = 1.0/1024.0
thresh_value = 2.1
chosen_seg = 2
xs = tf.placeholder(shape=[2,3,3], dtype=tf.float32)
sqnxs = ER._dot(xs, xs, keep_dims=True)
thresh = tf.constant(thresh_value * thresh_value/h)
# tensors = ER.TFSegmentVsSegmentDistanceFilter(h, 0, 2, xs, sqnxs, thresh, None)
# tensors = ER.TFSegmentVsSegmentDistanceFilter(h, 0, 1, xs, sqnxs, thresh, None)
# tensors = ER.TFSegmentVsSegmentDistanceFilter(h, 0, 1, xs, sqnxs, thresh, None)
init_stacked = -1 * tf.ones(shape=[1,4], dtype=tf.int64)
tensors = ER.TFSegmentVsSegmentDistanceFilterNoInnerCross(h, 1, 2, xs, sqnxs, thresh, init_stacked)
stacked = ER.TFSpecificSegmentDistanceFilter(h, np.int64(chosen_seg), xs, sqnxs, thresh, init_stacked)
complete = ER.TFDistanceFilter(h, xs, thresh)

xsd = np.array([
    [[-1, 0, 0],[0,0,0],[1,0,0]],
    [[-1, -1, 1],[-1,0,1],[-1,1,1]]
    ])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run(tensors, feed_dict={xs:xsd})
    for r in results:
        print(r)
    stacked_results = sess.run(stacked, feed_dict={xs:xsd})
    print(stacked_results)
    print_exp(xsd, thresh_value, chosen_seg)
    #for r in stacked_results:
        ##print(r)
    complete_results = sess.run(complete, feed_dict={xs:xsd})
    print(complete_results)
    print_exp(xsd, thresh_value, chosen_seg=None)
