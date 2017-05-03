#!/usr/bin/env python2
'''
Test for Motions
'''

import numpy as np
import scipy.io as spio
from ElasticRod import *
import ElasticRod as ER
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
    srod.InitTF(nrod) # TFRodCollisionImpulse requires tans for srod
    sela = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    selb = tf.placeholder(shape=[None, 1], dtype=tf.int32)

    convexity = TFRodCCD(crod, nrod, srod, sela, selb)
    # convexity = tf.reshape(convexity, [1])
    # print(convexity)
    # convexity.set_shape([1])
    # sel = tf.where(tf.equal(convexity, True))
    impulse_with_sel = TFRodCollisionImpulse(h, crod, nrod, srod, sela, selb)
    impulse = impulse_with_sel[0]
    # ASelS = impulse_with_sel[1]
    # BSelS = impulse_with_sel[2]
    # impulsed_rods = TFApplyImpulse(h, nrod, ASelS, BSelS, impulse)
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
    xsv = np.array([
        [-0.5, 0.0,-0.5],
        [-0.5, 1.0, 0.5],
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

class FakeRods:

    def GetMidPointsTF(self):
        if self.midpoints is None:
            xs_i_1, xs_i = ER._diffslices(self.xs)
            self.midpoints = (xs_i_1 + xs_i) / 2.0
        return self.midpoints

    def __init__(self):
        self.xs = None
        self.midpoints = None

def run_test2():
    '''
    Test 2: parallel rods
    '''
    f154 = spio.loadmat('testdata_ccd6/154.mat')
    f155 = spio.loadmat('testdata_ccd6/155.mat')
    raw_shape = f154['cpos'].shape
    cpos0 = f154['cpos'].reshape(raw_shape[1:])
    cpos1 = f155['cpos'].reshape(raw_shape[1:])
    # print(cpos0)
    # print(cpos1)
    # cpos1[0,:,:] += np.array([0,0,-0.1])
    xsshape = cpos0.shape
    print(xsshape)
    crod = FakeRods()
    nrod = FakeRods()
    srod = FakeRods()
    crod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)
    nrod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)
    srod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)

    h = 1.0/1024.0
    n = xsshape[1] - 1
    sela = tf.constant(np.array([[0, i] for i in range(n)]), dtype=tf.int32)
    selb = tf.constant(np.array([[1, i] for i in range(n)]), dtype=tf.int32)
    sela2 = tf.constant(np.array([[0, i+1] for i in range(n-1)]), dtype=tf.int32)
    selb2 = tf.constant(np.array([[1, i] for i in range(n-1)]), dtype=tf.int32)
    sela3 = tf.constant(np.array([[0, i-1] for i in range(1,n)]), dtype=tf.int32)
    selb3 = tf.constant(np.array([[1, i] for i in range(1,n)]), dtype=tf.int32)
    sela4 = tf.constant(np.array([[0, 7], [0, 7], [1, 6]]), dtype=tf.int32)
    selb4 = tf.constant(np.array([[1, 6], [1, 7], [0, 7]]), dtype=tf.int32)
    sela5 = tf.constant(np.array([[1, 6]]), dtype=tf.int32)
    selb5 = tf.constant(np.array([[0, 7]]), dtype=tf.int32)
    dpairs = TFDistanceFilter(h, crod.GetMidPointsTF(), tf.constant(20.0))
    dpairs_loose = TFDistanceFilter(h, crod.GetMidPointsTF(), tf.constant(200.0))
    sela_gcan = tf.slice(dpairs_loose, [0, 0], [-1, 2])
    selb_gcan = tf.slice(dpairs_loose, [0, 2], [-1, 2])
    convexity = TFRodCCDExtended(crod, nrod, srod, sela_gcan, selb_gcan)
    print(convexity.get_shape())
    fconvexity = ER._paddim(tf.cast(convexity, dtype=tf.int64))
    convex_sel = tf.concat([sela_gcan, selb_gcan, fconvexity], axis=1)

    inputdict={
            crod.xs:cpos0,
            nrod.xs:cpos1,
            srod.xs:cpos1,
            }

    cvx5 = TFRodCCD(crod, nrod, srod, sela5, selb5)
    cvx5_i = TFRodCCD(crod, nrod, srod, selb5, sela5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('{} expecting True '.format(sess.run(cvx5, feed_dict=inputdict)))
        print('{} expecting True '.format(sess.run(cvx5_i, feed_dict=inputdict)))
        # print(np.array(sess.run(srod.faceconvexity, feed_dict=inputdict)))
        # print(np.array(sess.run(srod.accumdots, feed_dict=inputdict)))
        print(sess.run(TFRodCCD(crod, nrod, srod, sela, selb), feed_dict=inputdict))
        print(sess.run(TFRodCCD(crod, nrod, srod, sela2, selb2), feed_dict=inputdict))
        print(sess.run(TFRodCCD(crod, nrod, srod, sela3, selb3), feed_dict=inputdict))
        print(sess.run(TFRodCCD(crod, nrod, srod, sela4, selb4), feed_dict=inputdict))
        print(sess.run(dpairs, feed_dict=inputdict))
        print(sess.run(convex_sel, feed_dict=inputdict))

def check_failure(frames, path):
    fdata = []
    for frame in frames:
        mat=spio.loadmat('{}/{}.mat'.format(path, frame))
        raw_shape = mat['cpos'].shape
        cpos = mat['cpos'].reshape(raw_shape[1:])
        fdata.append(cpos)
    # print(cpos0)
    # print(cpos1)
    # cpos1[0,:,:] += np.array([0,0,-0.1])
    xsshape = fdata[0].shape
    print(xsshape)
    crod = FakeRods()
    nrod = FakeRods()
    srod = FakeRods()
    crod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)
    nrod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)
    srod.xs = tf.placeholder(dtype=tf.float32, shape=xsshape)

    h = 1.0/1024.0
    n = xsshape[1] - 1
    dpairs = TFDistanceFilter(h, crod.GetMidPointsTF(), tf.constant(2400.0))
    sela_gcan = tf.slice(dpairs, [0, 0], [-1, 2])
    selb_gcan = tf.slice(dpairs, [0, 2], [-1, 2])
    sv1op = TFSignedVolumes(crod.xs, sela_gcan, selb_gcan)
    sv2op = TFSignedVolumes(nrod.xs, sela_gcan, selb_gcan)
    convexity = TFRodCCDExtended(crod, nrod, srod, sela_gcan, selb_gcan)
    collisions = ConvexityFilter(dpairs, convexity)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for f in range(len(frames)-1):
            cpos0 = fdata[f]
            cpos1 = fdata[f+1]
            inputdict={
                    crod.xs:cpos0,
                    nrod.xs:cpos1,
                    srod.xs:cpos1,
                    }
            cancol = sess.run(dpairs, feed_dict=inputdict)
            col = sess.run(collisions, feed_dict=inputdict)
            sv1,sv2 = sess.run([sv1op,sv2op], feed_dict=inputdict)
            print('frame {}, cancollision {} collision {}'.format(frames[f], cancol, col))
            print('SV1 {}'.format(sv1))
            print('SV2 {}'.format(sv2))
            print(cpos0-cpos1)
            print('crod {}'.format(sess.run(crod.xs, feed_dict=inputdict)))
            print('nrod {}'.format(sess.run(nrod.xs, feed_dict=inputdict)))

def run_test3():
    check_failure([47,48,49,51,51], 'testdata_ccd7')

def run_test4():
    check_failure([59,60,61,62], 'testdata_ccd7_CH2')

def run_test5():
    check_failure([132,133], 'testdata_ccd7_lowstiff_lowlearingrate')

def run_test6():
    check_failure([44,45,46], 'testdata_ccd8')

def run():
    run_test0()
    run_test1()
    run_test2()
    run_test3()
    run_test4()
    run_test5()
    run_test6()

if __name__ == '__main__':
    run_test6()
