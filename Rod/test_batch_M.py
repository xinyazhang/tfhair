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

def run_with_bc(n_rods, n_segs, h, rho, icond, path):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    irod = helper.create_TFRodS(n_rods, n_segs)
    irod.clone_args_from(icond)

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)

    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = 720
        # nframe = 10
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(nframe):
                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                # print(inputdict)
                saver.add_timestep(
                    [icond.xs],
                    [icond.thetas],
                    [icond.refd1s],
                    [icond.refd2s])
                xs, xdots, thetas, omegas = sess.run([orod.xs, orod.xdots,
                   orod.thetas, orod.omegas], feed_dict=inputdict)
                # print(orod.XForce.eval(feed_dict=inputdict))
                # print("xdots {}".format(xdots))
                # print("thetas {}".format(icond.thetas))
                icond = rrod.Relax(sess, irod, icond)
                # print("refd1s {}".format(icond.refd1s))
                # print("refd2s {}".format(icond.refd2s))
                progress.update(frame+1)

    saver.close()

def run_test0():
    '''
    Test 0: bending force only
    '''
    n_rods = 2
    n_segs = 2
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [-1,0,0], [0,0,0], [0,-1,0] ],
        [ [-1,0,1], [0,0,1], [0,-1,1] ],
    ])
    xdots = np.array([
        [ [0,0,0], [0,0,0], [0,0,0] ],
        [ [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest0b')

def run_test1():
    '''
    Test 1: bending force only
    '''
    n_rods = 2
    n_segs = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [-1,0,0], [0,0,0], [0,-1,0], [-1,-1,0] ],
        [ [-1,0,1], [0,0,1], [0,-1,1], [-1,-1,1] ],
    ])
    xdots = np.array([
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest1b')

def run_test2():
    '''
    Test 2: constant velocity
    '''
    n_rods = 2
    n_segs = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [-1,0,0], [0,0,0], [1,0,0], [2,0,0] ],
        [ [-1,0,1], [0,0,1], [1,0,1], [2,0,1] ],
    ])
    xdots = np.array([
        [ [1,10,0], [1,10,0], [1,10,0], [1,10,0] ],
        [ [1,10,0], [1,10,0], [1,10,0], [1,10,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest2b')

def run_test3():
    '''
    Test 3: twisting
    '''
    n_rods = 2
    n_segs = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [-1,0,0], [0,0,0], [1,0,0], [2,0,0] ],
        [ [-1,0,1], [0,0,1], [1,0,1], [2,0,1] ],
    ])
    xdots = np.array([
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.array([
        [0, pi, -pi],
        [0, pi, -pi],
    ])
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest3b')

def run_test4():
    '''
    Test 4: twisting with bending
    '''
    n_rods = 2
    n_segs = 2
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [0,0,0], [1,0,0], [1,-1,0] ],
        [ [0,0,1], [1,0,1], [1,-1,1] ],
    ])
    xdots = np.array([
        [ [0,0,0], [0,0,0], [0,0,0] ],
        [ [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.array([
        [0, 2 * pi],
        [0, 2 * pi],
    ])
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest4b')

def run_test5():
    '''
    Test 5: Constraints
    '''
    n_rods = 2
    n_segs = 5
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0] ],
        [ [0,0,1], [1,0,1], [2,0,1], [3,0,1], [4,0,1], [5,0,1] ],
    ])
    xdots = np.array([
        [ [240,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
        [ [240,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=initd1
    )
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest5b')

def run_test6():
    '''
    Test 6: bending force with gravity
    '''
    n_rods = 2
    n_segs = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [ [-1,0,0], [0,0,0], [0,-1,0], [-1,-1,0] ],
        [ [-1,0,1], [0,0,1], [0,-1,1], [-1,-1,1] ],
    ])
    xdots = np.array([
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
        [ [0,0,0], [0,0,0], [0,0,0], [0,0,0] ],
    ])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    thetas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    omegas = np.zeros(shape=[n_rods, n_segs], dtype=np.float32)
    icond = helper.create_BCRodS(xs=xs,
        xdots=xdots,
        thetas=thetas,
        omegas=omegas,
        initd1=initd1
    )
    icond.g = 9.8
    icond.floor_z = -5.0
    run_with_bc(n_rods, n_segs, h, rho, icond, '/tmp/tftest6b')

def run():
    # run_test0()
    run_test1()
    run_test2()
    run_test3()
    run_test4()
    run_test5()

if __name__ == '__main__':
    run()
