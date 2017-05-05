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
    if icond.ccd_threshold is None:
        icond.ccd_threshold = 30.0
    tf.reset_default_graph()
    irod = helper.create_TFRodS(2, n)
    irod.clone_args_from(icond)
    if icond.sparse_anchor_indices is not None:
        irod.sparse_anchor_indices = tf.placeholder(tf.int32, shape=[None, 2])
        irod.sparse_anchor_values = tf.placeholder(tf.float32, shape=[None, 3])

    orod = irod.CalcNextRod(h)
    rrod = orod.CalcPenaltyRelaxationTF(h)
    ''' Hack: This check collision b/w Rod 0 Seg # and Rod 1 Seg # '''
    # rrod.sela = tf.constant(np.array([[0, i] for i in range(n)]), dtype=tf.int32)
    # rrod.selb = tf.constant(np.array([[1, i] for i in range(n)]), dtype=tf.int32)
    rrod = rrod.CreateCCDNode(irod, h)

    # TODO: Calulate SelS in ElasticRodS directly.
    # sela_data = np.array([[0, i] for i in range(n)])
    # selb_data = np.array([[1, i] for i in range(n)])
# pfe = TFGetEConstaint(irod)
    saver = helper.RodSaver(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nframe = 720
        # nframe = 120
        with progressbar.ProgressBar(max_value=nframe) as progress:
            for frame in range(nframe):
                if icond_updater is not None:
                    icond_updater(h, icond)
                #inputdict = {irod.xs:xs, irod.restl:rl, irod.thetas:thetas, irod.xdots:xdots, irod:omegas:omegas}
                inputdict = helper.create_dict([irod], [icond])
                # inputdict.update({rrod.sela: sela_data, rrod.selb: selb_data})
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
                icond = rrod.Relax(sess, irod, icond, ccd_h=h, ccd_broadthresh=icond.ccd_threshold)
                # print('xs {}'.format(icond.xs))
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

def run_test4():
    '''
    Test 4: Multiple segments
    '''
    n = 10
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = helper.create_string(np.array([0,0.0,-2.25]), np.array([0,0.0,2.75]), n)
    rodb_xs = helper.create_string(np.array([1,-2.25,0.0]), np.array([1,2.75,0.0]), n)
    rods_xs = np.array([rodb_xs, roda_xs]) # B First
    # print(rods_xs)
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rodb_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    rods_xdots[0,:,0] = -10.0
    print(rods_xdots)
    initd1 = np.array([
        [1,0,0],
        [-1,0,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    icond.alpha = 0.05
    icond.beta = 0.05
    # icond.constraint_tolerance = 1 # low-stiffness rods
    # icond.anchor_stiffness = 1e3 # but we need maintain the anchor constrants
    # icond.t = 0.0
    run_with_bc(n, h, rho, icond, '/tmp/tfccd4')

def run_test5():
    '''
    Test 5: 0-degree Crossing
    '''
    n = 1
    h = 1.0/1024.0
    rho = 1.0

    roda_xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    rodb_xs = np.array([
        [-1,1.0,0],
        [0, 1.0,0],
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
        [0,-5,0],
        [0,-5,0],
        ])
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    run_with_bc(n, h, rho, icond, '/tmp/tfccd5')


def run_test6():
    '''
    Test 6: Twisting strings
    '''
    n = 60
    h = 1.0/1024.0
    rho = 1.0

    delta = (1.0/n)/2.0
    roda_xs = helper.create_string(np.array([0,0,-2.5 + delta]), np.array([0,0,2.5 + delta]), n)
    rodb_xs = helper.create_string(np.array([1,0.01,-2.5]), np.array([1,0.01,2.5]), n)
    rods_xs = np.array([rodb_xs, roda_xs]) # B First
    # print(rods_xs)
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rodb_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rods_xdots = np.array([roda_xdots, rodb_xdots])
    initd1 = np.array([
        [0,1,0],
        [0,1,0],
    ])
    icond = helper.create_BCRodS(xs=rods_xs,
            xdots=rods_xdots,
            thetas=rods_thetas,
            omegas=rods_thetas,
            initd1=initd1
            )
    icond.alpha = 0.05
    icond.beta = 0.05
    icond.constraint_tolerance = 5 # low-stiffness rods
    icond.anchor_stiffness = 1e3 # but we need maintain the anchor constrants
    icond.t = 0.0
    icond.ccd_threshold = 0.5/h
    def DualRotator(h, icond):
        #icond.sparse_anchor_values[2, :] = np.array([math.cos(icond.t), math.sin(icond.t), 5.0], dtype=np.float32)
        icond.sparse_anchor_values = np.array([
                [math.cos(icond.t), math.sin(icond.t)+0.01, -2.5 + icond.t * 0.005],
                [1, 0.01, 2.5],
                # [math.cos(-icond.t), math.sin(-icond.t)+0.01, 2.5],
                [0,0,-2.5 + delta],
                [0,0,2.5 + delta],
                ], dtype=np.float32)
        # print([math.cos(icond.t), math.sin(icond.t), 5.0])
        # print(icond.sparse_anchor_values)
        # print(icond.t)
        #icond.sparse_anchor_values[3] = np.array([math.cos(-icond.t), math.sin(-icond.t), 0.0 + icond.t])
        icond.t += h * 64
    icond.sparse_anchor_indices = np.array([
            [0, 0],
            [0, n],
            [1, 0],
            [1, n],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            rods_xs[0,0,:],
            rods_xs[0,-1,:],
            rods_xs[1,0,:],
            rods_xs[1,-1,:],
        ], dtype=np.float32)
    # icond.g = 9.8
    run_with_bc(n, h, rho, icond, '/tmp/tfccd6', icond_updater=DualRotator)

def run_test7():
    '''
    Test 7: Falling string
    '''
    n = 20
    h = 1.0/1024.0
    rho = 1.0
    height = 3.0

    delta = 0.2
    roda_xs = helper.create_string(np.array([-5,0,0]), np.array([5,0,0]), n)
    rodb_xs = helper.create_string(np.array([delta,-5+delta,height]), np.array([delta, 5+delta, height]), n)
    rods_xs = np.array([roda_xs, rodb_xs])
    # print(rods_xs)
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rodb_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
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
    icond.alpha = 0.5
    icond.beta = 0.05
    icond.t = 0.0
    icond.ccd_threshold = 5.0 / h
    icond.constraint_tolerance = 1e-5 * n * 2 # mid-stiffness rods
    icond.anchor_stiffness = 1e3 # maintain the anchor constrants
    icond.sparse_anchor_indices = np.array([
            [0, 0],
            [0, n],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            rods_xs[0,0,:],
            rods_xs[0,-1,:],
        ], dtype=np.float32)
    icond.g = 9.8
    run_with_bc(n, h, rho, icond, '/tmp/tfccd7')

def run_test8():
    '''
    Test 8: Falling segment
    Failure case: delta = -0.6
    '''
    n = 1
    h = 1.0/1024.0
    rho = 1.0
    height = 2.0

    delta = 0.0 #-1.6
    roda_xs = helper.create_string(np.array([-1+delta,0,0]), np.array([1 + delta,0,0]), n)
    rodb_xs = helper.create_string(np.array([0,-1,height]), np.array([0,1,height]), n)
    rods_xs = np.array([roda_xs, rodb_xs])
    # print(rods_xs)
    roda_thetas = np.zeros(shape=[n], dtype=np.float32)
    rodb_thetas = np.zeros(shape=[n], dtype=np.float32)
    rods_thetas = np.array([roda_thetas, rodb_thetas])
    roda_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
    rodb_xdots = np.zeros(shape=[n+1,3], dtype=np.float32)
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
    icond.alpha = 0.05
    icond.beta = 0.05
    icond.t = 0.0
    icond.ccd_threshold = 200.0 / h
    icond.sparse_anchor_indices = np.array([
            [0, 0],
            [0, n],
        ], dtype=np.int32)
    icond.sparse_anchor_values = np.array([
            rods_xs[0,0,:],
            rods_xs[0,-1,:],
        ], dtype=np.float32)
    icond.g = 9.8
    run_with_bc(n, h, rho, icond, '/tmp/tfccd8')

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

if __name__ == '__main__':
    run_test8()
