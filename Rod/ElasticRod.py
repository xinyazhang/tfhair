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

class ElasticRod:
    '''
    Convention of essential ElasticRod members
    xs: 2D (n+2) x 3 tensor, for vertex positions
    thetas: 1D n+1 tensor, for twisting on edges w.r.t. reference frame (aka Bishop frame)
            Note, we assumes piece-wise constant twisting
    refd1s, refd2s: 2D (n+1) x 3 tensor, to store reference directions
    '''

    xs = None
    thetas = None
    restl = None

    tan = None
    refd1s = None
    refd2s = None

    c = None # Translation
    q = None # Rotation quaternion

    def __init__(self, verts, restlength, thetas):
        self.xs = verts
        self.restl = restlength
        self.thetas = thetas
        '''
        self.multiplier should be initialized outside
        '''
        #self.multiplier = tf.Variable(tf.zeros(verts.get_shape()), dtype=tf.float32)  # constraint multipler

    '''
    Convention of additional tensors
    evec: 2D (n+1) x 3 tensor, for edge vectors (evec_i = x_{i+1} - x_i)
    ls: 1D n tensor with real elements, for voronoi region of each vertex (besides the first vertex)
    '''

'''
These functions assumes ElasticRod object with placeholders/tensors members.
'''

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
    '''
    shape = norms.shape.as_list()
    start = list([0] * len(shape))
    end = list(shape)
    end[dim] = shape[dim] - 1
    en_i_1 = tf.slice(norms, start, end)
    start[dim] = 1
    end[dim] = -1
    en_i = tf.slice(norms, start, end)
    '''
    en_i_1 = _trimslices(norms, dim, [0, 1])
    en_i = _trimslices(norms, dim, [1, 0])
    return en_i_1, en_i

def _normalize(evec):
    norm = tf.sqrt(_dot(evec, evec))
    norm = tf.stack([norm], _lastdim(norm)+1)
    print(evec.get_shape())
    print(norm.get_shape())
    return evec/norm

def TFGetEdgeVector(xs):
    x_i_1, x_i = _diffslices(xs, 0)
    return x_i - x_i_1

def TFGetEdgeLength(evec):
    norms = tf.norm(evec, axis=1)
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

# Calculate Intermediate Tensors (e.g. Curvatures) from Input Placeholders
def TFInitRod(rod):
    rod.evec = TFGetEdgeVector(rod.xs)
    rod.tan = _normalize(rod.evec)
#    rod.enorms = TFGetEdgeLength(rod.evec)
    rod.fullrestvl = TFGetVoronoiEdgeLength(rod.restl)
    rod.innerrestvl = _trimslices(rod.fullrestvl, 0, [1, 1])
    rod.ks = TFGetCurvature(rod.evec, rod.restl)
    return rod

# For unit \alpha
def TFGetEBend(rod):
    #return tf.reduce_sum(tf.multiply(tf.norm(rod.ks, ord=2, axis=1), 1.0/rod.restvl))
    sqnorm = tf.reduce_sum(tf.multiply(rod.ks, rod.ks), 1, keep_dims=False)
    return tf.reduce_sum(tf.multiply(sqnorm, 1.0/rod.innerrestvl))

def TFParallelTransportQuaternion(prod, crod):
    axes = tf.cross(prod.tan, crod.tan)
    cosines = _dot(prod.tan, crod.tan)
    halfconsines = tf.sqrt((cosines + 1)/2.0)
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
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    row1 = tf.stack([aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], axis=lastdim + 1)
    row2 = tf.stack([2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], axis=lastdim + 1)
    row3 = tf.stack([2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc], axis=lastdim + 1)
    R = tf.stack([row1, row2, row3], axis=lastdim + 2)
    refd1s = tf.stack([prod.refd1s], axis=_lastdim(prod.refd1s)+1)
    refd2s = tf.stack([prod.refd2s], axis=_lastdim(prod.refd2s)+1)
    shape = prod.refd1s.get_shape()
    crod.refd1s = tf.matmul(R, refd1s).reshape(shape)
    crod.refd2s = tf.matmul(R, refd2s).reshape(shape)
    return crod

# For unit \beta
# This should be a function w.r.t. thetas and xs
# which means difftheta also depends on xs
def TFGetETwist(rod):
    #theta_i_1, theta_i = _diffslices(rod.thetas)
    refd1primes = tf.cross(rod.ks, rod.refd1s)
    rod.mbars = tf.reduce_sum(tf.multiply(refd1primes, rod.refd2s), 1, keep_dims=False)
    difftheta = rod.thetas - rod.mbars
    return tf.reduce_sum(tf.multiply(difftheta*difftheta, 1.0/rod.innerrestvl))

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

# Calculate inextensibility constraints, i.e. rods non-stretchable
def TFGetCLength(rod):
    norms = TFGetEdgeLength(rod.evec)
    norms_sq = tf.reduce_sum(norms * norms)
    restl_sq = tf.reduce_sum(rod.restl * rod.restl)
    return norms_sq - restl_sq
