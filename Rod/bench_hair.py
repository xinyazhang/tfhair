#!/usr/bin/env python2
'''
Test for Hair Motions
'''

import sys
import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import scipy.io
import math
from math import pi
import progressbar
from tensorflow.python.client import timeline

def run_with_bc(n_rods, n_segs, h, rho, icond, anchors, path):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    icond.anchors = anchors[0]
    tf.reset_default_graph()
    irod = helper.create_TFRodS(n_rods, n_segs)
    irod.clone_args_from(icond)
    if icond.anchors is not None:
        irod.anchors = tf.placeholder(tf.float32, shape=icond.anchors.shape)

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)

    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        nframe = len(anchors)
        # nframe = 10
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(1,nframe):
                icond.anchors = anchors[frame-1]  # anchors update

                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                # print(inputdict)
                saver.add_timestep(
                    [icond.xs],
                    [icond.thetas],
                    [icond.refd1s],
                    [icond.refd2s])
                xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                   orod.thetas, orod.omegas], feed_dict=inputdict, options=run_options, run_metadata=run_metadata)
                # print(pfe.eval(feed_dict=inputdict))
                # print(orod.XForce.eval(feed_dict=inputdict))
                # print("xdots {}".format(xdots))
                # print("thetas {}".format(icond.thetas))
                icond = rrod.Relax(sess, irod, icond, options=run_options, run_metadata=run_metadata)
                # print("refd1s {}".format(icond.refd1s))
                # print("refd2s {}".format(icond.refd2s))
                progress.update(frame+1)
                # print icond.xs[:,0,:]
                # print icond.anchors
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('/tmp/bench_hair.json', 'w') as f:
            f.write(ctf)

    saver.close()

def run_hair_test(matfile):
    mdict = {}
    scipy.io.loadmat(matfile, mdict)

    xs = mdict["cpos"]
    xdots = mdict["cvel"]
    thetas = mdict["theta"]
    omegas = mdict["omega"]
    initd1 = mdict["initd"]
    anchors = mdict["anchor"]       # anchors records the anchor points of each frame
    icond = helper.create_BCRodS(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=initd1
    )
    icond.alpha = 0.001
    icond.beta = 0.001
    icond.g = 9.8

    n_rods, n_segs = thetas.shape
    h = 1.0/1024.0
    rho = 1.0

    run_with_bc(n_rods, n_segs, h, rho, icond, anchors, '/tmp/tfhair')

if __name__ == "__main__":
    run_hair_test(sys.argv[1])
