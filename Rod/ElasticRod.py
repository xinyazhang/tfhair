#!/usr/bin/env python2

import tensorflow as tf

'''
Elastic Rod Class

This is a very flexible class. It may store:
    1. The initial condition for each time step
    2. The final condition of current time step
    3. The placeholders and tensors to optimizing computational graph.
'''

class ElasticRod:
    '''
    Convention of essential ElasticRod members
    xs: 1D (n+2) x 3 tensor, for vertex positions
    thetas: 1D n+1 tensor, for twisting
    '''

    xs = None
    thetas = None
    restl = None

    c = None # Translation
    q = None # Rotation quaternion

    def __init__(self, verts, restlength, thetas):
        self.xs = verts
        self.restl = restlength
        self.thetas = thetas

    '''
    Convention of additional tensors
    evec: 2D (n+1) x 3 tensor, for edge vectors (evec_i = x_{i+1} - x_i)
    ls: 1D n tensor with real elements, for voronoi region of each vertex (besides the first vertex)
    '''

'''
These functions assumes ElasticRod object with placeholders/tensors members.
'''

def _diffslices(norms, dim = 0):
    shape = norms.shape.as_list()
    start = list([0] * len(shape))
    end = list(shape)
    end[dim] = shape[dim] - 1
    en_i_1 = tf.slice(norms, start, end)
    start[dim] = 1
    end[dim] = -1
    en_i = tf.slice(norms, start, end)
    return en_i_1, en_i

def TFGetEdgeVector(xs):
    shape = xs.shape.as_list()
    x_i_1, x_i = _diffslices(xs, 0)
    return x_i - x_i_1

def TFGetEdgeLength(evec):
    norms = tf.norm(evec, axis=1)
    return tf.reshape(norms, [evec.shape.as_list()[0]])

def TFGetVoronoiEdgeLength(enorms):
    en_i_1, en_i = _diffslices(enorms)
    return (en_i_1 + en_i) / 2

def TFGetCurvature(ev, enorms):
    e_i_1, e_i = _diffslices(ev, 0)
    en_i_1, en_i = _diffslices(enorms, 0)
    denominator1 = en_i_1 * en_i
    denominator2 = tf.reduce_sum(tf.multiply(e_i_1, e_i), 1, keep_dims=False)
    print(denominator2.shape)
    denominator = (denominator1+denominator2)
    shape3 = denominator.shape.as_list()
    denominator = tf.reshape(denominator, [shape3[0],1])
    return 2 * tf.multiply(tf.cross(e_i_1, e_i), 1.0/(denominator))

# For unit \alpha
def TFGetEBend(rod):
    rod.evec = TFGetEdgeVector(rod.xs)
#    rod.enorms = TFGetEdgeLength(rod.evec)
    rod.restvl = TFGetVoronoiEdgeLength(rod.restl)
    rod.ks = TFGetCurvature(rod.evec, rod.restl)
    #return tf.reduce_sum(tf.multiply(tf.norm(rod.ks, ord=2, axis=1), 1.0/rod.restvl))
    sqnorm = tf.reduce_sum(tf.multiply(rod.ks, rod.ks), 1, keep_dims=False)
    return tf.reduce_sum(tf.multiply(sqnorm, 1.0/rod.restvl))

# For unit \beta
def TFGetETwist(rod):
    theta_i_1, theta_i = _diffslices(rod.thetas)
    difftheta = theta_i - theta_i_1
    return tf.reduce_sum(tf.multiply(difftheta*difftheta, 1.0/rod.restvl))

# For unit \rho
def TFKinetic(rodnow, rodnext, h):
    xdot = 1/h * (rodnext.xs - rodnow.xs)
