#!/usr/bin/env python2
'''
Potential Energy
'''

import numpy as np
from ElasticRod import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi

def run_with_bc(n, h, rho, icond, path):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    irod = helper.create_TFRod(n)

    orod = irod.CalcNextRod(h)
    pfe = TFGetEConstaint(irod)
    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        for frame in range(720):
            #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
            inputdict = helper.create_dict([irod], [icond])
            # print(inputdict)
            paddedthetas = np.append(icond.thetas, 0.0)
            saver.add_timestep([icond.xs], [paddedthetas])
            xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                orod.thetas, orod.omegas], feed_dict=inputdict)
            # print(pfe.eval(feed_dict=inputdict))
            # print("thetas {}".format(thetas))
            # print("xdots {}".format(xdots))
            icond.xs = xs
            icond.xdots = xdots
            icond.thetas = thetas
            icond.omegas = omegas
    saver.close()

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
    h = 1.0/1024.0 * 4
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
    h = 1.0/1024.0 * 4
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

def run():
    run_test4()
    return
    run_test1()
    run_test2()
    run_test3()
    run_test4()

if __name__ == '__main__':
    run()
