#!/usr/bin/env python2

import tensorflow as tf

'''
Elastic Rod Class

This is a very flexible class. It may store:
    1. The initial condition for each time step
    2. The final condition of current time step
    3. The placeholders and tensors to optimizing computational graph.
'''

_epsilon = 1e-8

def _lastdim(tensor):
    return len(tensor.get_shape().as_list()) - 1

def _dot(tensor1, tensor2):
    dim = _lastdim(tensor1)
    return tf.reduce_sum(tf.multiply(tensor1, tensor2), dim, keep_dims=False)

def _trimslices(tensor, dim = 0, margins = [1, 1]):
    shape = tensor.get_shape().as_list()
    start = list([0] * len(shape))
    size = list(shape)
    start[dim] = margins[0]
    size[dim] -= margins[1] + margins[0]
    return tf.slice(tensor, start, size)

def _diffslices(norms, dim = 0):
    en_i_1 = _trimslices(norms, dim, [0, 1])
    en_i = _trimslices(norms, dim, [1, 0])
    return en_i_1, en_i

def _normalize(evec):
    norm = tf.sqrt(_dot(evec, evec))
    norm = tf.stack([norm], _lastdim(norm)+1)
    #print(evec.get_shape())
    #print(norm.get_shape())
    return evec/norm

def _paddim(tensor):
    return tf.stack([tensor], axis=_lastdim(tensor)+1)

def TFGetEdgeVector(xs):
    x_i_1, x_i = _diffslices(xs, 0)
    return x_i - x_i_1

def TFGetEdgeLength(evec):
    norms = tf.norm(evec, axis=_lastdim(evec))
    return tf.reshape(norms, [evec.get_shape().as_list()[0]])

def TFGetVoronoiEdgeLength(enorms):
    en_i_1, en_i = _diffslices(enorms)
    en_i_1 = tf.pad(enorms, [[0, 1]], 'CONSTANT')
    en_i = tf.pad(enorms, [[1, 0]], 'CONSTANT')
    return (en_i_1 + en_i) / 2

def TFGetCurvature(ev, enorms):
    e_i_1, e_i = _diffslices(ev, 0)
    en_i_1, en_i = _diffslices(enorms, 0)
    denominator1 = en_i_1 * en_i
    denominator2 = tf.reduce_sum(tf.multiply(e_i_1, e_i), 1, keep_dims=False)
    # print("TFGetCurvature: {}".format(denominator2.get_shape()))
    denominator = (denominator1+denominator2)
    shape3 = denominator.get_shape().as_list()
    denominator = tf.reshape(denominator, [shape3[0],1])
    return 2 * tf.multiply(tf.cross(e_i_1, e_i), 1.0/(denominator))

def TFGetLengthConstaintFunction(rod):
    sqlen = _dot(rod.evec, rod.evec)
    sqrest = rod.restl * rod.restl;
    return sqlen - sqrest

# For unit \alpha
def TFParallelTransportQuaternion(prod, crod):
    axes = tf.cross(prod.tans, crod.tans)
    cosines = _dot(prod.tans, crod.tans)
    halfconsines = tf.sqrt((cosines + 1)/2.0)
    halfconsines = tf.stack([halfconsines], _lastdim(halfconsines)+1) # pad one dimension
    #print(axes.get_shape())
    #print(halfconsines.get_shape())
    bcd = tf.multiply(axes, 0.5/halfconsines)
    return halfconsines, bcd

def TFPropogateRefDs(prod, crod):
    '''
    Calculate the current reference directions from the previous time frame.
    '''
    a, bcd = TFParallelTransportQuaternion(prod, crod)
    lastdim = _lastdim(bcd)
    margins = [[0,2],[1,1],[2,0]]
    b,c,d = map(lambda margin: _trimslices(bcd, _lastdim(bcd), margin), margins)
    # print(a.get_shape())
    # print(b.get_shape())
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    # print('aa shape {}'.format(aa.get_shape()))
    row1 = tf.concat([aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], axis=lastdim)
    # print('row1 shape {}, lastdim {}'.format(row1.get_shape(), lastdim))
    row2 = tf.concat([2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], axis=lastdim)
    row3 = tf.concat([2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc], axis=lastdim)
    R = tf.stack([row1, row2, row3], axis=lastdim)
    shape = prod.refd1s.get_shape()
    # print('prod.refd1s shape {}'.format(shape))
    refd1s = tf.stack([prod.refd1s], axis=_lastdim(prod.refd1s)+1)
    refd2s = tf.stack([prod.refd2s], axis=_lastdim(prod.refd2s)+1)
    crod.refd1s = tf.reshape(tf.matmul(R, refd1s), shape)
    crod.refd2s = tf.reshape(tf.matmul(R, refd2s), shape)
    return crod

class ElasticRod:

    '''
    Convention of essential ElasticRod members
    xs, xdots: 2D (n+1) x 3 tensor, for vertex positions/velocities
    restl: 1D (n) x 3 tensor
    thetas, omegas: 1D n tensor, for twisting/angular velocity on edges w.r.t. reference frame (aka Bishop frame)
            Note, we assumes PIECE-WISE CONSTANT twisting
    tans, refd1s, refd2s: 2D n x 3 tensor, to store reference directions
    '''

    xs = None
    thetas = None
    restl = None
    omegas = None

    tans = None
    refd1s = None
    refd2s = None

    rho = 1.0

    c = None # Translation
    q = None # Rotation quaternion

    '''
    Convention of additional tensors
    evec: 2D (n+1) x 3 tensor, for edge vectors (evec_i = x_{i+1} - x_i)
    ls: 1D n tensor with real elements, for voronoi region of each vertex (besides the first vertex)
    '''

    @staticmethod
    def CreateInputRod(n, rho = 1.0):
        return ElasticRod(
                xs=tf.placeholder(tf.float32, shape=[n+1, 3], name='xs'),
                restl=tf.placeholder(tf.float32, shape=[n], name='restl'),
                xdots=tf.placeholder(tf.float32, shape=[n+1, 3], name='xdots'),
                thetas=tf.placeholder(tf.float32, shape=[n], name='thetas'),
                omegas=tf.placeholder(tf.float32, shape=[n], name='omegas'),
                refd1s=tf.placeholder(tf.float32, shape=[n, 3], name='refd1s'),
                refd2s=tf.placeholder(tf.float32, shape=[n, 3], name='refd2s'),
                rho=rho)

    def __init__(self, xs, restl, xdots, thetas, omegas, refd1s = None, refd2s = None, rho = 1.0):
        self.xs = xs
        self.restl = restl
        self.xdots = xdots
        self.thetas = thetas
        self.omegas = omegas
        self.refd1s = refd1s
        self.refd2s = refd2s
        self.rho = rho

    def CalcNextRod(self, h):
        self.InitTF()
        E = self.GetEBendTF() + self.GetETwistTF()
        XForce, TForce = tf.gradients(-E, [self.xs, self.thetas])
        nxs = self.xs + h * self.xdots
        ndots = self.xdots + h * XForce / _paddim(self.fullrestvl * self.rho)
        nthetas = self.thetas + h * self.omegas
        nomegas = self.omegas + h * TForce / (self.restl * self.rho)
        nrod = ElasticRod(
                xs=nxs,
                restl=self.restl,
                xdots=ndots,
                thetas=nthetas,
                omegas=nomegas,
                rho=self.rho)
        nrod.InitTF() # FIXME(optimization): nrod only needs tans to TFPropogateRefDs
        if (not self.refd1s is None) and (not self.refd2s is None):
            TFPropogateRefDs(self, nrod)
        return nrod

    '''
    Functions with 'TF' suffix assume ElasticRod object members are tensors.
    '''
    def InitTF(rod):
        '''
        Calculate Intermediate Tensors (e.g. Curvatures) from Input Placeholders
        This is mandantory for Energy Terms
        '''
        rod.evec = TFGetEdgeVector(rod.xs)
        rod.tans = _normalize(rod.evec)
        rod.fullrestvl = TFGetVoronoiEdgeLength(rod.restl)
        rod.innerrestvl = _trimslices(rod.fullrestvl, _lastdim(rod.fullrestvl) - 1, [1, 1])
        rod.ks = TFGetCurvature(rod.evec, rod.restl)
        return rod

    def GetEConstaintTF(rod):
        diff =  TFGetLengthConstaintFunction(rod)
        return tf.reduce_sum(_dot(diff, diff))

    def GetEBendTF(rod):
        sqnorm = _dot(rod.ks, rod.ks)
        return tf.reduce_sum(tf.multiply(sqnorm, 1.0/rod.innerrestvl))

    def GetETwistTF(rod):
        '''
        For unit beta
        This should be a function w.r.t. thetas and xs
        which means difftheta also depends on xs
        '''
        #theta_i_1, theta_i = _diffslices(rod.thetas)
        refd1primes = tf.cross(rod.ks, _trimslices(rod.refd1s, margins=[0,1]))
        rod.mbars = _dot(refd1primes, _trimslices(rod.refd2s, margins=[0,1]))
        theta_i_1, theta_i = _diffslices(rod.thetas)
        difftheta = theta_i - theta_i_1
        deltatheta = difftheta - rod.mbars
        return tf.reduce_sum(tf.multiply(deltatheta*deltatheta, 1.0/rod.innerrestvl))

    def TFKineticI(rodnow, rodnext, h):
        rodnow.xdots = 1/h * (rodnext.xs - rodnow.xs)
        return TFKineticD(rodnow)

    def TFKineticD(rod):
        '''
        Calculate Kinetic Energy Directly from rod.xdots, For unit \rho
        '''
        xdot = rod.xdots
        xdot_i_1, xdot_i = _diffslices(xdot, 0)
        avexdot = 0.5 * (xdot_i_1 + xdot_i)
        sqnorm = tf.reduce_sum(tf.multiply(avexdot, avexdot), 1, keep_dims=False)
        return 0.5 * tf.reduce_sum(rod.restl * sqnorm)

def TFInitRod(rod):
    return rod.InitTF()

def TFGetEConstaint(rod):
    return rod.GetEConstaintTF()

def TFGetEBend(rod):
    return rod.GetEBendTF()

def TFGetETwist(rod):
    return rod.GetETwistTF()

# Calculate Kinetic Energy Indirectly, For unit \rho
def TFKineticI(rodnow, rodnext, h):
    rodnow.xdots = 1/h * (rodnext.xs - rodnow.xs)
    return TFKineticD(rodnow)

# Calculate Kinetic Energy Directly from rod.xdots, For unit \rho
def TFKineticD(rod):
    xdot = rod.xdots
    xdot_i_1, xdot_i = _diffslices(xdot, 0)
    avexdot = 0.5 * (xdot_i_1 + xdot_i)
    sqnorm = tf.reduce_sum(tf.multiply(avexdot, avexdot), 1, keep_dims=False)
    return 0.5 * tf.reduce_sum(rod.restl * sqnorm)

