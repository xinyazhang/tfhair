#!/usr/bin/env python2
'''
Test for Motions
'''

import numpy as np
from ElasticRod import *
from Obstacle import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi
import progressbar

def run_with_bc(n, h, rho, icond, path, icond_updater=None, obstacle=None):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    irod = helper.create_TFRod(n)
    irod.clone_args_from(icond)
    if icond.anchors is not None:
        irod.anchors = tf.placeholder(tf.float32, shape=icond.anchors.shape)
    if icond.sparse_anchor_indices is not None:
        irod.sparse_anchor_indices = tf.placeholder(tf.int32, shape=[None, 1])
        irod.sparse_anchor_values = tf.placeholder(tf.float32, shape=[None, 3])

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)
    if obstacle is not None:
        rrod.obstacle_impulse_op = obstacle.DetectAndApplyImpulseOp(h, rrod)

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
                spheres = None
                if obstacle is not None:
                    spheres = obstacle.GetSpheres()
                saver.add_timestep(
                    [icond.xs],
                    [icond.thetas],
                    [icond.refd1s],
                    [icond.refd2s],
                    spheres)
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
    icond = helper.create_BCRodS(xs=xs,
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

def run_test6():
    '''
    Test 6: bending force with gravity
    '''
    n = 3
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([
        [-1,0,  5],
        [0,0,   5],
        [0,-1,  5],
        [-1,-1, 5],
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
    icond.g = 9.8
    icond.floor_z = -5.0
    run_with_bc(n, h, rho, icond, '/tmp/tftest6')

def SinAnchor(h, icond):
    icond.anchors = np.array([0.0, math.sin(icond.t), 0.0])
    icond.t += h * 8

def run_test7():
    '''
    Test 7: anchors
    '''
    n = 32
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([[float(i)/(n / 4.0), 0.0, 0.0] for i in range(n+1)])
    xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    icond.anchors = np.array([0,0,0])
    icond.t = 0.0
    icond.alpha = 0.01
    icond.beta = 0.01
    run_with_bc(n, h, rho, icond, '/tmp/tftest7', icond_updater=SinAnchor)

def run_test8():
    '''
    Test 8: collision with unit sphere
    '''
    centers = [ np.array([0.0, 0.0, 0.0]) ]
    radii = [ np.array([1.0]) ]
    obstacle = SphericalBodyS(centers, radii)

    iterator = xrange(-5,6)
    n = len(list(iterator)) - 1
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([[i*0.5,-2,0] for i in iterator])
    xdots = np.array([[0,100,0]] * (n+1))
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0]))
    icond.alpha = 0.01
    icond.beta = 0.01

    run_with_bc(n, h, rho, icond, '/tmp/tftest8', obstacle=obstacle)

def FixedAnchor(h, icond):
    icond.anchors = np.array([0.0, 0.0, 1.0])

def run_test9():
    '''
    Test 9: collision with anchors on unit sphere
    '''
    centers = [ np.array([0.0, 0.0, 0.0]) ]
    radii = [ np.array([1.0]) ]
    obstacle = SphericalBodyS(centers, radii)

    iterator = xrange(1,11)
    n = len(list(iterator)) - 1
    h = 1.0/1024.0
    rho = 1.0

    xs = np.array([[0,0,i] for i in iterator])
    xdots = np.array([[0,0,0]] * n + [[0,100,0]])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0]))
    icond.alpha = 0.01
    icond.beta = 0.01

    icond.anchors = np.array([0,0,1])
    icond.g = 9.8

    run_with_bc(n, h, rho, icond, '/tmp/tftest9', icond_updater=FixedAnchor, obstacle=obstacle)

def run_test10():
    '''
    Test 10: ahoge, hair with strong bending coefficient,
    dangling effect
    This requires 2000 iterations
    '''
    centers = [ np.array([0.0, 0.0, 0.0]) ]
    radii = [ np.array([1.0]) ]
    obstacle = SphericalBodyS(centers, radii)

    iterator = xrange(1,6)
    n = len(list(iterator)) - 1
    h = 1.0/1024.0
    rho = 1e-2

    xs = np.array(
        [[0,0,i] for i in iterator]
    )
    xdots = np.array(
        [[0,0,0]] * n + [[0,10,0]]
    )
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    initd1 = np.array(
        [0,1,0]
    )
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=initd1)
    icond.alpha = 100
    icond.beta = 0.01

    icond.sparse_anchor_indices = np.array([
            [0],
            [1],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            xs[0,:],
            xs[1,:],
        ], dtype=np.float32)
    icond.g = 9.8

    run_with_bc(n, h, rho, icond, '/tmp/tftest10', obstacle=obstacle)

def run_test11():
    '''
    Test 11: twisting with anchor points
    '''

    iterator = xrange(-5,6)
    n = len(list(iterator)) - 1
    h = 1.0/1024.0
    rho = 1e-2

    def FixedTwister(h, icond):
        icond.thetas[0] = 0
        icond.thetas[-1] = pi * icond.t
        icond.t += h * 8

    xs = np.array(
        [[i,0,0] for i in iterator]
    )
    xdots = np.array(
        [[0,0,0]] * (n+1)
    )
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    icond.sparse_anchor_indices = np.array([
            [0],
            [2],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            xs[0,:],
            xs[2,:],
        ], dtype=np.float32)
    icond.beta = 10
    icond.t = 0
    run_with_bc(n, h, rho, icond, '/tmp/tftest11', icond_updater=FixedTwister)

def run_test12():
    '''
    Test 12: twisting with anchor points, version 2
    '''

    iterator1 = xrange(-1,0)
    iterator2 = xrange(0,2)
    n1 = len(list(iterator1))
    n2 = len(list(iterator2))
    n = n1 + n2 - 1
    h = 1.0/1024.0
    rho = 1e-2

    def FixedTwisterII(h, icond):
        icond.thetas[0] = pi * icond.t
        icond.xs[0] = np.array([-n1, 0, 0])
        icond.xs[n1] = np.array([0, 0, 0])
        icond.t += h * 8

    xs = np.array(
        [[i,0,0] for i in iterator1] +
        [[0,i,0] for i in iterator2]
    )
    xdots = np.array(
        [[0,0,0]] * (n+1)
    )
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    icond = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    icond.sparse_anchor_indices = np.array([
            [0],
            [2],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            xs[0,:],
            xs[2,:],
        ], dtype=np.float32)
    icond.beta = 10
    icond.t = 0
    run_with_bc(n, h, rho, icond, '/tmp/tftest12', icond_updater=FixedTwisterII)

def run():
    run_test0()
    run_test1()
    run_test2()
    run_test3()
    run_test4()
    run_test5()
    run_test6()
    run_test7()
    run_test8()
    run_test9()
    run_test10()
    run_test11()
    run_test12()

if __name__ == '__main__':
    run()
