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
import optparse

def run_with_bc(opt, n_rods, n_segs, h, rho, icond, anchors, path, obstacle=None):
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

    if opt.collision:
        rrod = rrod.CreateCCDNode(irod, h)

    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = len(anchors)
        # nframe = 10
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(1,nframe):
                icond.sparse_anchor_values = anchors[frame-1]  # anchors update
                inputdict = helper.create_dict([irod], [icond])
                saver.add_timestep(
                    [icond.xs],
                    [icond.thetas],
                    [icond.refd1s],
                    [icond.refd2s])
                xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                   orod.thetas, orod.omegas], feed_dict=inputdict)
                if opt.collision:
                    icond = rrod.Relax(sess, irod, icond, ccd_h=h, ccd_broadthresh=icond.ccd_threshold)
                else:
                    icond = rrod.Relax(sess, irod, icond)
                progress.update(frame+1)

    saver.close()

def run_hair_bench(matfile, options):
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
    icond.ccd_resolve_hard = False

    n_rods, n_segs = thetas.shape
    h = 1.0/1024.0
    rho = 1e-2

    icond.sparse_anchor_indices = anchor_indices
    icond.sparse_anchor_values = anchor_values[0]

    centers = mdict["obstacle_centers"]
    radii = mdict["obstacle_radii"]
    obstacle = SphericalBodyS(centers, radii)

    path = os.path.join("/tmp/tfhair", os.path.basename(matfile).split(".")[0])
    run_with_bc(options, n_rods, n_segs, h, rho, icond, anchor_values, path, obstacle=obstacle)

def parse_args():
    parser = optparse.OptionParser()
    parser.add_option("", "--enable-collision", dest="collision",
            action="store_true", default=False, help="enable collision, default to false")
    return parser.parse_args()

if __name__ == "__main__":
    options, args = parse_args()
    if len(args) == 0:
        print "Please specify the location of initial condition, i.e. mat file"
    if options.collision:
        print "Start hair simulation with collision"
    run_hair_bench(args[0], options)
