#!/usr/bin/env python2
'''
Test for Motions
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # suppress tensorflow warnings

import numpy as np
from ElasticRod import *
from Obstacle import *
import RodHelper as helper
import tensorflow as tf
import math
from math import pi
import progressbar
import optparse

all_tests = dict()

def run_with_bc(n, h, rho, icond, path, icond_updater=None, obstacle=None):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    ''' Create ElasticRodS with tf.placeholder(s) as the input '''
    irod = helper.create_TFRod(n)
    ''' Copy arguments and constraints from input conditions '''
    irod.clone_args_from(icond)
    if icond.anchors is not None:
        irod.anchors = tf.placeholder(tf.float32, shape=icond.anchors.shape)
    if icond.sparse_anchor_indices is not None:
        irod.sparse_anchor_indices = tf.placeholder(tf.int32, shape=[None, 1])
        irod.sparse_anchor_values = tf.placeholder(tf.float32, shape=[None, 3])

    ''' Create the ElasticRodS for unconstrainted positions '''
    orod = irod.CalcNextRod(h)
    ''' Create the ElasticRodS for constrainted positions '''
    rrod = orod.CalcPenaltyRelaxationTF(h)
    if obstacle is not None:
        ''' Setup constraint operators'''
        rrod.obstacle_impulse_op = obstacle.DetectAndApplyImpulseOp(h, rrod)

    saver = helper.RodSaver(path)
    ''' Create a TF session for actual computation '''
    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            nframe = 720
            # nframe = 10
            with progressbar.ProgressBar(max_value=nframe) as progress:
                for frame in range(nframe):
                    if icond_updater is not None:
                        icond_updater(h, icond)
                    inputdict = helper.create_dict([irod], [icond])
                    spheres = None
                    if obstacle is not None:
                        spheres = obstacle.GetSpheres()
                    saver.add_timestep(
                        [icond.xs],
                        [icond.thetas],
                        [icond.refd1s],
                        [icond.refd2s],
                        spheres)
                    icond = rrod.Relax(sess, irod, icond)
                    progress.update(frame+1)
    except Exception as e:
        print(e)

    saver.close()

def test(test_func):
    def test_wrapper():
        print "=> Running {test}".format(test=test_func.__name__)
        test_func()
    all_tests[test_func.__name__] = test_wrapper
    return test_wrapper

@test
def run_test0():
    '''
    Test 0: bending force only
    '''
    n = 2
    h = 1.0/1024.0
    rho = 1.0

    '''
    Setup the initial conditions and drop them to run_with_bc
    '''
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

@test
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

@test
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

@test
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

@test
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

@test
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

@test
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

@test
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

    def SinAnchor(h, icond):
        icond.anchors = np.array([0.0, math.sin(icond.t), 0.0])
        icond.t += h * 8

    run_with_bc(n, h, rho, icond, '/tmp/tftest7', icond_updater=SinAnchor)

@test
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

@test
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

    def FixedAnchor(h, icond):
        icond.anchors = np.array([0.0, 0.0, 1.0])

    run_with_bc(n, h, rho, icond, '/tmp/tftest9', icond_updater=FixedAnchor, obstacle=obstacle)

@test
def run_test10():
    '''
    Test 10: ahoge: top hair with strong bending
    coefficient, dangling effect
    Sometimes it requires 2000 iterations
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

@test
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

@test
def run_test12():
    '''
    Test 12: twisting with anchor points, version 2
    '''

    """
    works only for half_segs=1,2
    fails to correctly propagate on the third segment
    """
    half_segs = 3
    iterator1 = xrange(-half_segs,0)
    iterator2 = xrange(0,half_segs+1)
    n1 = len(list(iterator1))
    n2 = len(list(iterator2))
    n = n1 + n2 - 1
    h = 1.0/1024.0
    rho = 1e-4

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
            [half_segs],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            xs[0,:],
            xs[half_segs,:],
        ], dtype=np.float32)
    icond.anchor_stiffness = 1e4
    icond.alpha = 0.002
    icond.beta = 10
    icond.constraint_iterations = 2000
    icond.t = 0
    icond.rho = rho
    run_with_bc(n, h, rho, icond, '/tmp/tftest12', icond_updater=FixedTwisterII)

def parse_args():
    parser = optparse.OptionParser()
    parser.add_option("", "--tests", dest="tests",
            default=None, help="specify tests to run, separated by comma ','")
    return parser.parse_args()

if __name__ == '__main__':
    options, _ = parse_args()

    # select tests to run
    if options.tests is None or options.tests == "all":
        tests_to_run = sorted(all_tests.keys(), key=lambda x: int(filter(str.isdigit, x)))
    else:
        tests_to_run = map(lambda x : "run_test{number}".format(number=x), options.tests.split(','))

    # run tests
    for func_name in tests_to_run:
        if func_name in all_tests:
            all_tests[func_name]()
