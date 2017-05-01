#!/usr/bin/env python2

import os
import shutil
import scipy.io
import numpy as np
import ElasticRod
import tensorflow as tf
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def calculate_norms(e):
    return np.sum(np.abs(e)**2, axis=-1)**(1./2)

def divide_vecarray_by_scalararray(numerator, denominator):
    return np.divide(numerator.T, denominator).T

def normalize(e):
    return divide_vecarray_by_scalararray(e, calculate_norms(e))

def calculate_rest_length(xs):
    #nvert = xs.shape[0]
    e = xs[1:,:] - xs[0:-1,:]
    return calculate_norms(e)

def calculate_rest_lengths(xs):
    e = xs[:,1:,:] - xs[:,0:-1,:]
    return calculate_norms(e)

def calculate_kb(e):
    norms = np.sum(np.abs(e)**2, axis=-1)**(1./2)
    e_i_1 = e[0:-1,:]
    e_i = e[1:,:]
    kb_numerator = 2 * np.cross(e_i_1, e_i)
    kb_denominator = np.multiply(norms[:-1], norms[1:]) + np.einsum('ij,ij->i', e_i_1, e_i)
    kb_denominator.reshape([kb_denominator.shape[0],1])
    # print('kb_numerator {}'.format(kb_numerator))
    # print('kb_denominator {}'.format(kb_denominator))
    return divide_vecarray_by_scalararray(kb_numerator, kb_denominator)

def calculate_parallel_transport(eprev, ethis):
    vector = np.cross(eprev, ethis)
    norm = math.fabs(calculate_norms(vector))
    if norm < 1e-9:
        return np.identity(3)
    axis = vector
    cosine = np.dot(eprev, ethis)
    cosine = max(-1.0, min(1.0, cosine))    # sometimes there is 1+epsilon
    theta = math.acos(cosine)
    return rotation_matrix(axis, theta)

def calculate_reference_directions(xs, initd1):
    e = xs[1:,:] - xs[0:-1,:]
    ebar = normalize(e)
    kb = calculate_kb(e)
    # print('edge {}'.format(e))
    # print('kb {}'.format(kb))
    prevd1 = initd1
    prevd2 = np.cross(e[0], initd1)
    # print('initd1 {}'.format(prevd1))
    # print('initd2 {}'.format(prevd2))
    d1arr = [prevd1]
    d2arr = [prevd2]
    for i in range(1, e.shape[0]):
        P = calculate_parallel_transport(ebar[i-1], ebar[i])
        d1 = P.dot(prevd1)
        d2 = P.dot(prevd2)
        # print('d1[{}]: {}'.format(i, d1))
        # print('d2[{}]: {}'.format(i, d2))
        d1arr.append(normalize(d1))
        d2arr.append(normalize(d2))
        prevd1 = d1
        prevd2 = d2
    return np.array(d1arr), np.array(d2arr)

def calculate_batch_reference_directions(xs, initd1s):
    batch_refd1s = []
    batch_refd2s = []
    for i in range(xs.shape[0]):
        refd1s, refd2s = calculate_reference_directions(xs[i], initd1s[i])
        batch_refd1s.append(refd1s)
        batch_refd2s.append(refd2s)
    return np.array(batch_refd1s), np.array(batch_refd2s)

def create_TFRod(n_segs):
    return ElasticRod.ElasticRodS.CreateInputRod(n_segs)

def create_TFRodS(n_rods, n_segs):
    return ElasticRod.ElasticRodS.CreateInputRodS(n_rods, n_segs)

def create_BCRodS(xs, xdots, thetas, omegas, initd1):
    if len(xs.shape) == 2:
        rl = calculate_rest_length(xs)
        creator = calculate_reference_directions
    else:
        rl = calculate_rest_lengths(xs)
        creator = calculate_batch_reference_directions
    rod = ElasticRod.ElasticRodS(xs=xs, restl=rl, xdots=xdots, thetas=thetas, omegas=omegas)
    rod.refd1s, rod.refd2s = creator(xs, initd1)
    return rod

def create_BCRod(xs, xdots, thetas, omegas, initd1):
    return create_BCRodS(xs, xdots, thetas, omegas, initd1)

def create_dict(irods, drods):
    '''
    Create a dict by assiging placeholders in irod with values in drod
    '''
    nelem = len(irods)
    tups = []
    for i in range(nelem):
        irod = irods[i]
        drod = drods[i]
        #if type(irod.xs) is tf.placeholder:
        tups.append((irod.xs, drod.xs))
        tups.append((irod.restl, drod.restl))
        tups.append((irod.xdots, drod.xdots))
        tups.append((irod.thetas, drod.thetas))
        tups.append((irod.omegas, drod.omegas))
        # tups.append((irod.rho, drod.rho))
        if not irod.refd1s is None:
            tups.append((irod.refd1s, drod.refd1s))
        if not irod.refd2s is None:
            tups.append((irod.refd2s, drod.refd2s))
        if irod.anchors is not None:
            tups.append((irod.anchors, drod.anchors))
        if irod.sparse_anchor_indices is not None:
            tups.append((irod.sparse_anchor_indices, drod.sparse_anchor_indices))
            tups.append((irod.sparse_anchor_values, drod.sparse_anchor_values))

    return dict(tups)

def create_string(start, end, nseg):
    ret = [start]
    for i in range(1, nseg):
        r = float(i)/nseg
        ret.append((1-r)*start + r * end)
    ret.append(end)
    return np.array(ret)

class RodSaver():

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.frame = 0
        # clean dest directory
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass

    def add_timestep(self, cpos, thetas, refd1s, refd2s):
        filename = os.path.join(self.directory, str(self.frame))
        mdict = {
            "cpos"   : np.array(cpos),
            "thetas" : np.array(thetas),
            "refd1s" : np.array(refd1s),
            "refd2s" : np.array(refd2s)
        }
        scipy.io.savemat(filename, mdict, appendmat=True)
        self.frame += 1
