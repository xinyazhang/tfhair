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
    tf.reset_default_graph()
    irod = helper.create_TFRodS(2, n)
    irod.alpha = 0.01
    irod.beta = 0.01

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)
    rrod = rrod.CreateCCDNode(irod, h)

    # TODO: Calulate SelS in ElasticRodS directly.
    ''' This check collision b/w Rod 0 Seg # and Rod 1 Seg # '''
    sela_data = np.array([[0, i] for i in range(n)])
    selb_data = np.array([[1, i] for i in range(n)])

    # pfe = TFGetEConstaint(irod)
    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = 720
        # nframe = 120
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(nframe):
                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                inputdict.update({rrod.sela: sela_data, rrod.selb: selb_data})
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
                icond = rrod.Relax(sess, irod, icond, ccd_h=h, SelS=[sela_data, selb_data])
                # print("refd1s {}".format(icond.refd1s))
                # print("refd2s {}".format(icond.refd2s))
                progress.update(frame+1)

    saver.close()

def run_test0():
    '''
    Test 0: 90-degree Crossing, homogeneous mass
    '''
    n = 1
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    rodb_xs = np.array([
        [-0.5,0.5,-0.5],
        [-0.5,0.5,0.5],
        ])
    rods_xs = np.array([roda_xs, rodb_xs])
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.array([
        [0,5,0],
        [0,5,0],
        ])
    rodb_xdots = np.array([
        [0,0,0],
        [0,0,0],
        ])
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [1,0,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    run_with_bc(n, h, rho, icond, '/tmp/tfccd0')

def run_test1():
    '''
    Test 0: 90-degree Crossing
    '''
    n = 1
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    rodb_xs = np.array([
        [-0.5,0.5,-1],
        [-0.5,0.5,1],
        ])
    rods_xs = np.array([roda_xs, rodb_xs])
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.array([
        [0,5,0],
        [0,5,0],
        ])
    rodb_xdots = np.array([
        [0,0,0],
        [0,0,0],
        ])
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [1,0,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    run_with_bc(n, h, rho, icond, '/tmp/tfccd1')

def run_test2():
    '''
    Test 2: 45-degree Crossing
    '''
    n = 1
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    rodb_xs = np.array([
        [-0.5, 0.0,-0.5],
        [-0.5, 1.0, 0.5],
        ])
    rods_xs = np.array([roda_xs, rodb_xs])
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.array([
        [0,5,0],
        [0,5,0],
        ])
    rodb_xdots = np.array([
        [0,0,0],
        [0,0,0],
        ])
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [1,0,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    run_with_bc(n, h, rho, icond, '/tmp/tfccd2')

def run_test3():
    '''
    Test 3: Multiple segments
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = np.array([
        [-2,0,0],
        [-1,0,0],
        [0,0,0],
        [1,0,0],
        ])
    rodb_xs = np.array([
        [-2, 1,-1],
        [-1, 1, 1],
        [ 0, 1,-1],
        [ 1, 1, 1],
        ])
    rods_xs = np.array([roda_xs, rodb_xs])
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.array([
        [0,5,0],
        [0,5,0],
        [0,5,0],
        [0,5,0],
        ])
    rodb_xdots = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0],
        ])
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [1,0,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    run_with_bc(n, h, rho, icond, '/tmp/tfccd3')

def run():
    run_test0()
    run_test1()
    run_test2()
    run_test3()

if __name__ == '__main__':
    run()
