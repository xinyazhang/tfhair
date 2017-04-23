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

def run_with_bc(n, h, rho, icond, path):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    irod = helper.create_TFRod(n)

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)

    pfe = TFGetEConstaint(irod)
    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = 720
        with progressbar.ProgressBar(max_value=nframe-1) as progress:
            for frame in range(nframe):
                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                # print(inputdict)
                paddedthetas = np.append(icond.thetas, 0.0)
                saver.add_timestep([icond.xs], [paddedthetas])
                # xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                #    orod.thetas, orod.omegas], feed_dict=inputdict)
                # print(pfe.eval(feed_dict=inputdict))
                # print(orod.XForce.eval(feed_dict=inputdict))
                # print("xdots {}".format(xdots))
                # print("thetas {}".format(icond.thetas))
                icond = rrod.Relax(sess, irod, icond)
                progress.update(frame)

    saver.close()

def run_test0():
    '''
    Test 0: bending force only
    '''
    n = 2
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [0,-1,0],
        ])
    xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest0')

def run_test1():
    '''
    Test 1: bending force only
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [0,-1,0],
        [-1,-1,0],
        ])
    xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest1')

def run_test2():
    '''
    Test 2: constant velocity
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        [2,0,0],
        ])
    xdots = np.array([
        [1,10,0],
        [1,10,0],
        [1,10,0],
        [1,10,0],
        ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest2')

def run_test3():
    '''
    Test 3: twisting
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        [2,0,0],
        ])
    xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.array([0, pi, -pi])
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest3')

def run_test4():
    '''
    Test 4: twisting with bending
    '''
    n = 2
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [0,0,0],
        [1,0,0],
        [1,-1,0],
        ])
    xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.array([0, 2 * pi])
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest4')

def run_test5():
    '''
    Test 5: Constraints
    '''
    n = 5
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [0,0,0],
        [1,0,0],
        [2,0,0],
        [3,0,0],
        [4,0,0],
        [5,0,0],
        ])
    xdots = np.array([
        [240,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, rho, icond, '/tmp/tftest5')

def run():
    run_test0()
    run_test1()
    run_test2()
    run_test3()
    run_test4()
    run_test5()

if __name__ == '__main__':
    run()
