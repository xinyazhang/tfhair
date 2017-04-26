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

def run_with_bc(n, h, crod_data, nrod_data, srod_data):
    '''
    Run the simulation with given boundary conditions (icond)
    '''
    tf.reset_default_graph()
    crod = helper.create_TFRod(n)
    nrod = helper.create_TFRod(n)
    srod = helper.create_TFRod(n)
    srod.InitTF() # TFRodCollisionImpulse requires tans for srod
    sela = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    selb = tf.placeholder(shape=[None, 1], dtype=tf.int32)

    convexity = TFRodCCD(crod, nrod, srod, sela, selb)
    #convexity = tf.reshape(convexity, [1])
    # print(convexity)
    # convexity.set_shape([1])
    # sel = tf.where(tf.equal(convexity, True))
    impulse = TFRodCollisionImpulse(h, crod, nrod, srod, sela, selb, convexity)
    # seltest1 = TFRodXSel(crod, sela)
    sela_data = [[i] for i in range(len(crod_data.xs)-1)]
    selb_data = [[i] for i in range(len(srod_data.xs)-1)]
    # print('sela_data {}'.format(sela_data))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputdict = helper.create_dict([crod, nrod, srod], [crod_data, nrod_data, srod_data])
        inputdict.update({sela: sela_data, selb: selb_data})
        print(sess.run(convexity, inputdict))
        # print(sess.run(sel, inputdict))
        print('Impulse {}'.format(sess.run(impulse, inputdict)))
        # print(sess.run(seltest, inputdict))

def test_static_rods(xs, xs2, xsv, ccd_result):
    n = 1
    h = 1.0/1024.0
    rho = 1.0

    xdots = np.array([
        [0,0,0],
        [0,0,0],
        ])
    thetas = np.zeros(shape=[n], dtype=np.float32)
    omegas = np.zeros(shape=[n], dtype=np.float32)
    crod_data = helper.create_BCRod(xs=xs,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    nrod_data = helper.create_BCRod(xs=xs2,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    srod_data = helper.create_BCRod(xs=xsv,
            xdots=xdots,
            thetas=thetas,
            omegas=omegas,
            initd1=np.array([0,1,0])
            )
    run_with_bc(n, h, crod_data, nrod_data, srod_data)
    print('Expecting {}'.format(ccd_result))

def run_test0():
    '''
    Test 0: Crossing
    '''
    xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    xs2 = np.array([
        [-1,1,0],
        [0,1,0],
        ])
    xsv = np.array([
        [-0.5,0.5,-1],
        [-0.5,0.5,1],
        ])
    test_static_rods(xs, xs2, xsv, True)
    xsv = np.array([
        [-0.5,0.5,0.001],
        [-0.5,0.5,-0.001],
        ])
    test_static_rods(xs, xs2, xsv, True)
    xsv = np.array([
        [-0.5,0.5,0.0],
        [0.0,0.5,0.0],
        ])
    test_static_rods(xs, xs2, xsv, True)
    xsv = np.array([
        [-1.5,0.5,0.0],
        [0.0,0.5,0.0],
        ])
    test_static_rods(xs, xs2, xsv, True)

def run_test1():
    '''
    Test 1: Non-Crossing
    '''
    xs = np.array([
        [-1,0,0],
        [0,0,0],
        ])
    xs2 = np.array([
        [-1,1,0],
        [0,1,0],
        ])
    xsv = np.array([
        [-1.002,0.5,-1],
        [-1.002,0.5,1],
        ])
    test_static_rods(xs, xs2, xsv, False)
    xsv = np.array([
        [-0.5,0.5,0.001],
        [-0.5,0.5,1],
        ])
    test_static_rods(xs, xs2, xsv, False)
    xsv = np.array([
        [-0.5,0.5,-0.002],
        [-0.5,0.5,-0.001],
        ])
    test_static_rods(xs, xs2, xsv, False)
    xsv = np.array([
        [-0.5,0.5,0.001],
        [0.0,0.5,0.001],
        ])
    test_static_rods(xs, xs2, xsv, False)

def run_test2():
    '''
    Test 2: ConvexityFilter
    '''

def run():
    run_test0()
    run_test1()
    run_test2()

if __name__ == '__main__':
    run()
