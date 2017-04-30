
#!/usr/bin/env python2
'''
Test for Motions
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi
import progressbar

def run_with_bc(n, h, rho, icond, path, icond_updater=None):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    irod = helper.create_TFRod(n)
    irod.clone_args_from(icond)
    if icond.anchors is not None:
        irod.anchors = tf.placeholder(tf.float32, shape=icond.anchors.shape)
    if icond.body_collision is not None:
        irod.body_collision = icond.body_collision

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)

    # pfe = TFGetEConstaint(irod)
    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = 720
        # nframe = 10
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(nframe):
                if icond_updater is not None:
                    icond_updater(h, icond)
                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                # print(inputdict)
                saver.add_timestep(
                    [icond.xs],
                    [icond.thetas],
                    [icond.refd1s],
                    [icond.refd2s])
                # xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                #    orod.thetas, orod.omegas], feed_dict=inputdict)
                # print(pfe.eval(feed_dict=inputdict))
                # print(orod.XForce.eval(feed_dict=inputdict))
                # print("xdots {}".format(xdots))
                # print("thetas {}".format(icond.thetas))
                icond = rrod.Relax(sess, irod, icond)
                # print("refd1s {}".format(icond.refd1s))
                # print("refd2s {}".format(icond.refd2s))
                progress.update(frame+1)

    saver.close()

def CollisionCheck(rod):
    center = tf.Variable(np.array([0,0,0]), dtype=tf.float32, trainable=False)
    radius = tf.Variable(np.array([1.0]), dtype=tf.float32, trainable=False)
    return TKGetForbiddenSphere(center, radius, rod)

def run_test1():
    '''
    Test 1: collision with unit sphere
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([ [-1,-2,0], [0,-2,0], [1,-2,0], [2,-2,0], ])
    xdots = np.array([ [0,10,0], [0,10,0], [0,10,0], [0,10,0], ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0]))
    icond.alpha = 0.01
    icond.beta = 0.01

    icond.body_collision = CollisionCheck

    run_with_bc(n, h, rho, icond, '/tmp/tftest1')

def run():
    run_test1()

if __name__ == '__main__':
    run()
