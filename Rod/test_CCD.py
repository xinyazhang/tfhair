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

    def __init__(self, xs):
        self.xs = xs
        self.evec = TFGetEdgeVector(self.xs)
        self.tans = ER._normalize(self.evec)
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
    crod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))
    nrod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))
    srod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))

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

def check_failure(frames, path, narrow_distance=0.1, false_positive=False):
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
    crod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))
    nrod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))
    srod = FakeRods(tf.placeholder(dtype=tf.float32, shape=xsshape))

    h = 1.0/1024.0
    n = xsshape[1] - 1
    dpairs = TFDistanceFilter(h, crod.GetMidPointsTF(), tf.constant(narrow_distance/h))
    sela_gcan = tf.slice(dpairs, [0, 0], [-1, 2])
    selb_gcan = tf.slice(dpairs, [0, 2], [-1, 2])
    sv1op = TFSignedVolumes(crod.xs, sela_gcan, selb_gcan)
    sv2op = TFSignedVolumes(nrod.xs, sela_gcan, selb_gcan)
    convexity = TFRodCCDExtended(crod, nrod, srod, sela_gcan, selb_gcan)
    convexity_location = tf.where(tf.equal(convexity, True))
    collisions = CollisionFilter(dpairs, convexity)

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
            cvx = sess.run(convexity, feed_dict=inputdict)
            cvx_loc = sess.run(convexity_location, feed_dict=inputdict)
            if len(cvx_loc) == 0:
                if false_positive:
                    print("False positive PASSED")
                    continue
            else:
                cvx_loc = cvx_loc[0]
            col = sess.run(collisions, feed_dict=inputdict)
            sv1,sv2 = sess.run([sv1op,sv2op], feed_dict=inputdict)
            print("frame {}\ncancollision {}\nconvexity {}\ncollision {}".format(frames[f], cancol, cvx, col))
            print('SV1 {}'.format(sv1))
            print('SV2 {}'.format(sv2))
            print('CVX Location {}'.format(cvx_loc))
            if not false_positive:
                print('abcd {}'.format(sess.run([srod.dbg_a, srod.dbg_b, srod.dbg_c, srod.dbg_d], feed_dict=inputdict)))
                print('p {}'.format(sess.run(srod.dbg_p, feed_dict=inputdict)))
                print('q {}'.format(sess.run(srod.dbg_q, feed_dict=inputdict)))
                print('single real roots {}'.format(sess.run(srod.dbg_signleroots, feed_dict=inputdict)))
                print('    Abar {}'.format(sess.run(srod.dbg_Abar, feed_dict=inputdict)))
                print('    Phibar {}'.format(sess.run(srod.dbg_Phibar, feed_dict=inputdict)))
                print("triple real roots\n")
                print("\tA {}".format(sess.run(srod.dbg_A, feed_dict=inputdict)))
                print("\tPhi {}".format(sess.run(srod.dbg_Phi, feed_dict=inputdict)))
                print("\tB {}".format(sess.run(srod.dbg_B, feed_dict=inputdict)))
                print('tri real root1 {}'.format(sess.run(srod.dbg_triroots1, feed_dict=inputdict)))
                print('tri real root2 {}'.format(sess.run(srod.dbg_triroots2, feed_dict=inputdict)))
                print('tri real root3 {}'.format(sess.run(srod.dbg_triroots3, feed_dict=inputdict)))
                print('quadratic real roots1 {}'.format(sess.run(srod.dbg_quadroots1, feed_dict=inputdict)))
                print('quadratic real roots2 {}'.format(sess.run(srod.dbg_quadroots2, feed_dict=inputdict)))
                # print('roots2 {}'.format(sess.run(srod.dbg_roots2, feed_dict=inputdict)))
                print('s and t {}'.format(sess.run([srod.dbg_s, srod.dbg_t], feed_dict=inputdict)))
                print('linear roots {}'.format(sess.run(srod.dbg_linroot, feed_dict=inputdict)))
                print('linear valid {}'.format(sess.run(srod.dbg_linvalid, feed_dict=inputdict)))
                print('linear sgood {}'.format(sess.run(srod.dbg_sgood, feed_dict=inputdict)))
                print('linear tgood {}'.format(sess.run(srod.dbg_tgood, feed_dict=inputdict)))
                print('linear taugood {}'.format(sess.run(srod.dbg_taugood, feed_dict=inputdict)))
                print('linear rxsnz {}'.format(sess.run(srod.dbg_rxsnz, feed_dict=inputdict)))
                print('linear (q-p)xr {}'.format(sess.run(srod.dbg_q_pxr, feed_dict=inputdict)))
            else:
                sel=cvx_loc[0]
                a,b,c,d = sess.run([srod.dbg_a, srod.dbg_b, srod.dbg_c, srod.dbg_d], feed_dict=inputdict)
                rxs = sess.run(srod.dbg_rxs, feed_dict=inputdict)
                qroots1 = sess.run(srod.dbg_quadroots1, feed_dict=inputdict)
                qroots2 = sess.run(srod.dbg_quadroots2, feed_dict=inputdict)
                s,t = sess.run([srod.dbg_s, srod.dbg_t], feed_dict=inputdict)
                lroots = sess.run(srod.dbg_linroot, feed_dict=inputdict)
                linvalid = sess.run(srod.dbg_linvalid, feed_dict=inputdict)
                print("abcd {}".format([a[sel],b[sel],c[sel],d[sel]]))
                print("qroot1 {}".format(qroots1[sel]))
                print("qroot2 {}".format(qroots2[sel]))
                print("rxs {}".format(rxs[sel]))
                print("s,t {}{}".format(s[sel], t[sel]))
                print("lroot {}".format(lroots[sel]))
                print("lvalid {}".format(linvalid[sel]))

            '''
            # print(np.array(sess.run(srod.faceconvexity, feed_dict=inputdict))[:, s])
            print('cancollision {}'.format(cancol[s]))
            print('SV1 {}'.format(sv1[s]))
            print('SV2 {}'.format(sv2[s]))
            print('cpos fixed {}-{}'.format(cpos0[0,10], cpos0[0,11]))
            print('cpos moving {}-{}'.format(cpos0[1,9], cpos0[1,10]))
            print('npos fixed {}-{}'.format(cpos1[0,10], cpos1[0,11]))
            print('npos moving {}-{}'.format(cpos1[1,9], cpos1[1,10]))
            # print('npos {}'.format(sv2[s]))
            # print('crod {}'.format(sess.run(crod.xs, feed_dict=inputdict)))
            # print('nrod {}'.format(sess.run(nrod.xs, feed_dict=inputdict)))
            '''

def run_test3():
    check_failure([47,48,49,51,51], 'testdata_ccd7')

def run_test4():
    check_failure([59,60,61,62], 'testdata_ccd7_CH2')

def run_test5():
    check_failure([132,133], 'testdata_ccd7_lowstiff_lowlearingrate')

def run_test6():
    check_failure([44,45,46], 'testdata_ccd8')

def run_test7():
    check_failure([50,51], 'testdata_ccd7_cvx')

def run_test8():
    check_failure([140,141,142,143], 'testdata_ccd8_direct')

def run_test9():
    check_failure([117,118,119], 'testdata_ccd7_rccd')
    check_failure([123,124,125,126,127], 'testdata_ccd7_rccd')

def run_test10():
    check_failure([141,142,143], 'testdata_ccd7_rccd2')

def run_test11():
    check_failure([330,331,332], 'testdata_ccd7_rccd3')

def run_test12():
    check_failure([285,286,287], 'testdata_ccd7_rccd4')

def run_test13():
    check_failure([121,122,123], 'testdata_ccd7_rccd5')
    check_failure([152, 153], 'testdata_ccd7_rccd5')

def run_test14():
    check_failure([51,52], 'testdata_ccd6_rccd', 0.01)

def run_test15():
    check_failure([51,52], 'testdata_ccd6_rccd', 0.15)

def run_test16():
    ''' False positives, expecting no collisions '''
    check_failure([52,53], 'testdata_ccd6_rccd_fp', 0.15, false_positive=True)

def run_test17():
    check_failure([80,81,82], 'testdata_ccd6_rccd2', 0.0025)

def run_test18():
    check_failure([110,111,112], 'testdata_ccd6_rccd2', 0.0025)

def run_test19():
    check_failure([110,111,112], 'testdata_ccd6_rccd3', 0.0025)

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
    run_test13()
    run_test14()
    run_test15()
    run_test16()
    run_test17()
    run_test18()
    run_test19()

if __name__ == '__main__':
    run()
