#!/usr/bin/env python2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # suppress tensorflow warnings

import sys
import numpy as np
from ElasticRod import *
from Obstacle import *
import RodHelper as helper
import tensorflow as tf
import scipy.io
import math
from math import pi
import progressbar
from tensorflow.python.client import timeline

def run_with_bc(n_rods, n_segs, h, rho, icond, anchors, path, obstacle=None):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    irod = helper.create_TFRodS(n_rods, n_segs)
    irod.clone_args_from(icond)

    irod.sparse_anchor_indices = tf.placeholder(tf.int32, shape=[None, 2])
    irod.sparse_anchor_values = tf.placeholder(tf.float32, shape=[None, 3])

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h, 7e-5)
    # rrod = orod.CalcPenaltyRelaxationTF(h)
    if obstacle is not None:
        rrod.obstacle_impulse_op = obstacle.DetectAndApplyImpulseOp(h, rrod)

    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        nframe = len(anchors)
        # nframe = 10
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(1,nframe):
                icond.sparse_anchor_values = anchors[frame-1]  # anchors update

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

def HeadCollisionPotential(rod):
    center = tf.Variable(np.array([0,0,0]), dtype=tf.float32, trainable=False)
    radius = tf.Variable(np.array([1.0]), dtype=tf.float32, trainable=False)
    return TKGetForbiddenSphere(center, radius, rod)

def run_hair_bench(matfile):
    mdict = {}
    scipy.io.loadmat(matfile, mdict)

    xs = mdict["cpos"]
    xdots = mdict["cvel"]
    thetas = mdict["theta"]
    omegas = mdict["omega"]
    initd1 = mdict["initd"]
    anchor_values = mdict["anchor"]
    anchor_indices = mdict["anchor_indices"]
    icond = helper.create_BCRodS(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=initd1
    )
    icond.alpha = 0
    icond.beta = 0
    icond.g = 9.8

    n_rods, n_segs = thetas.shape
    h = 1.0/1024.0
    rho = 1e-2

    icond.sparse_anchor_indices = anchor_indices
    icond.sparse_anchor_values = anchor_values[0]

    centers = mdict["obstacle_centers"]
    radii = mdict["obstacle_radii"]
    obstacle = SphericalBodyS(centers, radii)

    path = os.path.join("/tmp/tfhair", os.path.basename(matfile).split(".")[0])
    run_with_bc(n_rods, n_segs, h, rho, icond, anchor_values, path, obstacle=obstacle)

if __name__ == "__main__":
    run_hair_bench(sys.argv[1])
