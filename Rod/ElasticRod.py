#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import RodHelper as helper
import math

'''
Elastic Rod Class

This is a very flexible class. It may store:
    1. The initial condition for each time step
    2. The final condition of current time step
    3. The placeholders and tensors to optimizing computational graph.
'''

_epsilon = 1e-8
_default_rho = 0.1
_stiff = 1e7

def _ndim(tensor):
    return len(tensor.get_shape())

def _lastdim(tensor):
    return _ndim(tensor) - 1

def _dot(tensor1, tensor2, dim=None):
    if dim == None:
        dim = _lastdim(tensor1)
    return tf.reduce_sum(tf.multiply(tensor1, tensor2), dim, keep_dims=False)

def _trimslices(tensor, dim = None, margins = [1, 1]):
    if dim == None:
        dim = _lastdim(tensor) - 1
    shape = tensor.get_shape()
    start = list([0] * len(shape))
    size = list(shape.as_list())
    start[dim] = margins[0]
    size[dim] -= margins[1] + margins[0]
    return tf.slice(tensor, start, size)

def _diffslices(norms, dim = None):
    if dim is None:
        dim = _lastdim(norms) - 1
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
    #return tf.expand_dims(tensor, -1)

def TFGetEdgeVector(xs):
    x_i_1, x_i = _diffslices(xs)
    return x_i - x_i_1

def TFGetEdgeLength(evec):
    norms = tf.norm(evec, axis=_lastdim(evec))
    return tf.reshape(norms, [evec.get_shape().as_list()[0]])

def TFGetVoronoiEdgeLength(enorms):
    en_i_1, en_i = _diffslices(enorms)
    # print(en_i_1)
    # print(len(en_i_1.get_shape()))
    paddings = [[0,0] for i in range(len(enorms.get_shape()))]
    # print(paddings)
    paddings[-1] = [0,1]
    # print(paddings)
    en_i_1 = tf.pad(enorms, paddings, 'CONSTANT')
    paddings[-1] = [1,0]
    # print(paddings)
    en_i   = tf.pad(enorms, paddings, 'CONSTANT')
    return (en_i_1 + en_i) / 2

def TFGetCurvature(ev, enorms):
    e_i_1, e_i = _diffslices(ev)
    en_i_1, en_i = _diffslices(enorms, dim=_lastdim(enorms))
    denominator1 = en_i_1 * en_i
    #denominator2 = tf.reduce_sum(tf.multiply(e_i_1, e_i), _lastdim(e_i), keep_dims=False)
    denominator2 = _dot(e_i_1, e_i)
    # print("ev: {}".format(ev.get_shape()))
    # print("enorms: {}".format(enorms))
    # print("e_i_1: {}".format(e_i_1.get_shape()))
    # print("en_i_1: {}".format(en_i_1.get_shape()))
    # print("TFGetCurvature 1: {}".format(denominator1.get_shape()))
    # print("TFGetCurvature 2: {}".format(denominator2.get_shape()))
    denominator = (denominator1+denominator2)
    shape3 = denominator.get_shape().as_list()
    # print(denominator.get_shape())
    denominator = _paddim(denominator)
    # print(denominator.get_shape())
    return 2 * tf.multiply(tf.cross(e_i_1, e_i), 1.0/(denominator))

def TFGetLengthConstraintFunction(rod):
    sqlen = _dot(rod.evec, rod.evec)
    sqrest = rod.restl * rod.restl;
    return sqlen - sqrest

# For unit \alpha
def TFParallelTransportQuaternion(prod, crod):
    axes = -tf.cross(prod.tans, crod.tans)
    cosines = _dot(prod.tans, crod.tans)
    halfconsines = tf.sqrt((cosines + 1)/2.0)
    halfconsines = tf.stack([halfconsines], _lastdim(halfconsines)+1) # pad one dimension
    #print(axes.get_shape())
    #print(halfconsines.get_shape())
    bcd = tf.multiply(axes, 0.5/halfconsines)
    return halfconsines, bcd

def TFPropogateRefDs(prod, crod, normalize=False):
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
    refd1s = _paddim(prod.refd1s)
    refd2s = _paddim(prod.refd2s)
    crod.refd1s = tf.reshape(tf.matmul(R, refd1s), shape)
    crod.refd2s = tf.reshape(tf.matmul(R, refd2s), shape)
    if normalize:
        crod.refd1s = _normalize(crod.refd1s)
        crod.refd2s = _normalize(crod.refd2s)
    return crod

def TFRodXSel(rod, Sel):
    xs_i_1, xs_i = _diffslices(rod.xs, dim=_lastdim(rod.xs)-1)
    gxs_i_1 = tf.gather_nd(xs_i_1, Sel)
    gxs_i = tf.gather_nd(xs_i, Sel)
    # FIXME: More general cases
    # gxs_i_1.set_shape([None, 3])
    # gxs_i.set_shape([None, 3])
    return gxs_i_1, gxs_i

def TFRodXDotSel(rod, Sel):
    xdots_i_1, xdots_i = _diffslices(rod.xdots, dim=_lastdim(rod.xdots)-1)
    gxdots_i_1 = tf.gather_nd(xdots_i_1, Sel)
    gxdots_i = tf.gather_nd(xdots_i, Sel)
    # FIXME: More general cases
    # gxdots_i_1.set_shape([None, 3])
    # gxdots_i.set_shape([None, 3])
    return (gxdots_i_1 + gxdots_i) / 2.0

def TFConvexityByList(tensormat):
    faceconvexity = []
    # print(tensormat[0][0].get_shape())
    # print('dim {}'.format(dim))
    for i in range(len(tensormat)):
        tensorlist = tensormat[i]
        axes = tf.cross(tensorlist[1] - tensorlist[0], tensorlist[2] - tensorlist[0])
        dotsigns = []
        for j in range(3, 6):
            delta = tensorlist[j] - tensorlist[0]
            dots = _dot(delta, axes)
            #dots = muls
            # dots.set_shape([None])
            signs = tf.sign(dots)
            dotsigns.append(signs)
            '''
            print('delta shape {}'.format(delta.get_shape()))
            print('axes shape {}'.format(axes.get_shape()))
            print('dots shape {}'.format(dots.get_shape()))
            print('signs shape {}'.format(signs.get_shape()))
            '''
        # print('dotsigns[0] shape: {}'.format(dotsigns[0].get_shape()))
        faceconvexity.append(tf.equal(dotsigns[0], dotsigns[1]))
        faceconvexity.append(tf.equal(dotsigns[0], dotsigns[2]))
    # return faceconvexity
    convexity = faceconvexity[0]
    # print('faceconvexity[0] shape: {}'.format(faceconvexity[0].get_shape()))
    for i in range(1, len(faceconvexity)):
        convexity = tf.logical_and(convexity, faceconvexity[i])
    # convexity.set_shape([None])
    return convexity

def TFRodCCDExtended(crod, nrod, srod, ASelS, BSelS):

    '''
    Continus Collision Detection between Rod A and Rod B

    crod: ElasticRod object representing current position of Rod A
    nrod: ElasticRod object representing next position of Rod A
    srod: ElasticRod object representing current position of Rod B
    ASelS, BSelS: Selecting Tensors for Rod A and B respectively
          Check tf.gather for more details
    Note: assumes 2 now, 4 may use TFRodSCCD
    '''
    ''' Gathered Current Xs '''
    gcxs_k_1, gcxs_k = TFRodXSel(crod, ASelS)
    # print(gcxs_k.get_shape())
    # return tf.shape(gcxs_k)
    ''' Gathered Next Xs '''
    gnxs_k_1, gnxs_k = TFRodXSel(nrod, ASelS)
    npoles, spoles = TFRodXSel(srod, BSelS) # FIXME: use srod based reference system
    verts = [
            [npoles, gcxs_k_1, gcxs_k, gnxs_k, gnxs_k_1, spoles],
            [npoles, gcxs_k, gnxs_k, gnxs_k_1, gcxs_k_1, spoles],
            [npoles, gnxs_k, gnxs_k_1, gcxs_k_1, gcxs_k, spoles],
            [npoles, gnxs_k_1, gcxs_k_1, gcxs_k, gnxs_k, spoles],
            [spoles, gcxs_k_1, gcxs_k, gnxs_k, gnxs_k_1, npoles],
            [spoles, gcxs_k, gnxs_k, gnxs_k_1, gcxs_k_1, npoles],
            [spoles, gnxs_k, gnxs_k_1, gcxs_k_1, gcxs_k, npoles],
            [spoles, gnxs_k_1, gcxs_k_1, gcxs_k, gnxs_k, npoles]
            ]
    # return [verts[0]]
    convexity = TFConvexityByList(verts)
    return convexity, gcxs_k_1, gcxs_k, gnxs_k_1, gnxs_k

def TFRodCCD(crod, nrod, srod, ASelS, BSelS):
    a_list = TFRodCCDExtended(crod, nrod, srod, ASelS, BSelS)
    return a_list[0]

def ConvexityFilter(SelS_in, convexity):
    return tf.gather_nd(SelS_in, tf.where(tf.equal(convexity, True)))

# TODO: merge with TFRodCCD
def TFRodCollisionImpulse(h, crod, nrod, srod, ASelS_in, BSelS_in, convexity = None):
    gcxs_k = None
    if convexity is None:
        convexity, _, _, _, _ = TFRodCCDExtended(crod, nrod, srod, ASelS_in, BSelS_in)
    ASelS = ConvexityFilter(ASelS_in, convexity)
    BSelS = ConvexityFilter(BSelS_in, convexity)

    ''' Gathered Current Xs '''
    gcxs_k_1, gcxs_k = TFRodXSel(crod, ASelS)
    ''' Gathered Next Xs '''
    gnxs_k_1, gnxs_k = TFRodXSel(nrod, ASelS)

    gqdots = ((gnxs_k_1 - gcxs_k_1) + (gnxs_k - gcxs_k)) / (2 * h)
    # return tf.shape(gnxs_k_1)
    # return tf.shape(gqdots)
    grqdots = TFRodXDotSel(srod, BSelS)
    relqdots = gqdots - grqdots
    gtans = tf.gather_nd(srod.tans, BSelS)
    # return relqdots
    # return gtans
    #gtans.set_shape([None, 3])
    gnormals = tf.cross(gtans, tf.cross(gtans, relqdots))
    gsegmass = _paddim(tf.gather_nd(nrod.restl, ASelS))
    # print('gmass: {}'.format(gmass))
    return 0.5 * h * gnormals * gsegmass, ASelS, BSelS

def TFApplyImpulse(h, rods, ASelS, BSelS, impulse):
    # TODO: apply impulse to Rod B
    # deltaVB = tf.SparseTensor(BSelS, -impulse)
    # delta = deltaVA + deltaVB
    # deltaVA = tf.SparseTensor(ASelS, impulse * 2, dense_shape=rods.xs.get_shape())
    # delta = deltaVA
    ASelX = ASelS
    BSelX = BSelS
    ASelX_p1 = ASelS + tf.constant([0,1])
    BSelX_p1 = BSelS + tf.constant([0,1])
    indices = tf.concat([ASelS, ASelX_p1 , BSelS, BSelX_p1], axis=0)
    gAvertmass = _paddim(tf.gather_nd(rods.fullrestvl, ASelX))
    gAvertmass_p1 = _paddim(tf.gather_nd(rods.fullrestvl, ASelX_p1))
    gBvertmass = _paddim(tf.gather_nd(rods.fullrestvl, BSelX))
    gBvertmass_p1 = _paddim(tf.gather_nd(rods.fullrestvl, BSelX_p1))
    impulse_to_A = impulse / gAvertmass
    impulse_to_A_p1 = impulse / gAvertmass_p1
    impulse_to_B = -impulse / gBvertmass
    impulse_to_B_p1 = -impulse / gBvertmass_p1
    updates = tf.concat([impulse_to_A, impulse_to_A_p1, impulse_to_B, impulse_to_B_p1], axis=0)
    return tf.scatter_nd_add(rods.xs, indices, updates)
    # nrods.xdots = tf.scatter_add(nrods.xdots, ASelS, impulse)
    # TFPropogateRefDs(rods, nrods, normalize=True)
    '''
    Note we don't need to change xdots or others, this is an optimizaiton procedure
    '''
    # return nrods

class ElasticRodS:

    '''
    Convention of essential ElasticRod members
    xs, xdots: 2D (n+1) x 3 tensor, for vertex positions/velocities
    restl: 1D (n) tensor
    thetas, omegas: 1D n tensor, for twisting/angular velocity on edges w.r.t. reference frame (aka Bishop frame)
            Note, we assumes PIECE-WISE CONSTANT twisting
    tans, refd1s, refd2s: 2D n x 3 tensor, to store reference directions

    ElasticRodS class represents a collection of rods.
    '''

    xs = None
    thetas = None
    restl = None
    omegas = None

    tans = None
    refd1s = None
    refd2s = None

    rho = _default_rho
    alpha = 1.0
    beta = 1.0
    g = 0.0
    floor_z = -50.0

    def clone_args_from(self, other):
        if other is None:
            return self
        self.rho = other.rho
        self.alpha = other.alpha
        self.beta = other.beta
        self.g = other.g
        self.floor_z = other.floor_z
        self.anchors = other.anchors
        self.anchor_masks = other.anchor_masks
        return self

    c = None # Translation
    q = None # Rotation quaternion
    anchors = None # 2D tensor: N Rod x 3
    anchor_masks = None # 2D tensor: N Rod x 1

    '''
    Convention of additional tensors
    evec: 2D (n+1) x 3 tensor, for edge vectors (evec_i = x_{i+1} - x_i)
    ls: 1D n tensor with real elements, for voronoi region of each vertex (besides the first vertex)
    '''

    @staticmethod
    def CreateInputRod(n_segs, rho = _default_rho):
        return ElasticRodS(
                xs=tf.placeholder(tf.float32, shape=[n_segs+1, 3], name='xs'),
                restl=tf.placeholder(tf.float32, shape=[n_segs], name='restl'),
                xdots=tf.placeholder(tf.float32, shape=[n_segs+1, 3], name='xdots'),
                thetas=tf.placeholder(tf.float32, shape=[n_segs], name='thetas'),
                omegas=tf.placeholder(tf.float32, shape=[n_segs], name='omegas'),
                refd1s=tf.placeholder(tf.float32, shape=[n_segs, 3], name='refd1s'),
                refd2s=tf.placeholder(tf.float32, shape=[n_segs, 3], name='refd2s'),
                rho=rho)

    @staticmethod
    def CreateInputRodS(n_rods, n_segs, rho = _default_rho):
        return ElasticRodS(
                xs=tf.placeholder(tf.float32, shape=[n_rods, n_segs+1, 3], name='xs'),
                restl=tf.placeholder(tf.float32, shape=[n_rods, n_segs], name='restl'),
                xdots=tf.placeholder(tf.float32, shape=[n_rods, n_segs+1, 3], name='xdots'),
                thetas=tf.placeholder(tf.float32, shape=[n_rods, n_segs], name='thetas'),
                omegas=tf.placeholder(tf.float32, shape=[n_rods, n_segs], name='omegas'),
                refd1s=tf.placeholder(tf.float32, shape=[n_rods, n_segs, 3], name='refd1s'),
                refd2s=tf.placeholder(tf.float32, shape=[n_rods, n_segs, 3], name='refd2s'),
                rho=rho)

    def __init__(self, xs, restl, xdots, thetas, omegas, refd1s = None, refd2s = None, rho = _default_rho):
        self.xs = xs
        self.restl = restl
        self.xdots = xdots
        self.thetas = thetas
        self.omegas = omegas
        self.refd1s = refd1s
        self.refd2s = refd2s
        self.rho = rho

    def CalcNextRod(self, h):
        self.InitTF(None) # FIXME(optimization): only needs tans for TFPropogateRefDs
        nxs = self.xs + h * self.xdots
        nthetas = self.thetas + h * self.omegas
        pseudonrod = ElasticRodS(
                xs=nxs,
                restl=self.restl,
                xdots=self.xdots,
                thetas=nthetas,
                omegas=self.omegas,
                rho=self.rho)
        pseudonrod.InitTF(self)
        if (not self.refd1s is None) and (not self.refd2s is None):
            TFPropogateRefDs(self, pseudonrod)
        E = self.alpha * pseudonrod.GetEBendTF() \
                + self.beta * pseudonrod.GetETwistTF() \
                + pseudonrod.GetEGravityTF() \
                # + _stiff * pseudonrod.GetEConstaintTF()
        # print('E: {}'.format(E))
        # print('pseudonrod.xs: {}'.format(pseudonrod.xs))
        # print('pseudonrod.thetas: {}'.format(pseudonrod.thetas))
        # TForce = tf.gradients(-E, pseudonrod.thetas)
        # XForce = tf.gradients(-E, pseudonrod.xs)
        XForce, TForce = tf.gradients(-E, [pseudonrod.xs, pseudonrod.thetas])
        # print('XForce: {}'.format(XForce))
        # print('TForce: {}'.format(TForce))
        nxdots = self.xdots + h * XForce / _paddim(self.fullrestvl * self.rho)
        nomegas = self.omegas + h * TForce / (self.restl * self.rho)
        nrod = ElasticRodS(
                xs=nxs,
                restl=self.restl,
                xdots=nxdots,
                thetas=nthetas,
                omegas=nomegas,
                refd1s=pseudonrod.refd1s,
                refd2s=pseudonrod.refd2s,
                rho=self.rho)
        nrod.XForce = XForce
        nrod.TForce = TForce
        nrod.InitTF(pseudonrod)
        return nrod

    def CalcPenaltyRelaxationTF(self, h, learning_rate=1e-4):
        xs = tf.Variable(np.zeros(shape=self.xs.get_shape().as_list(), dtype=np.float32), name='xs')
        relaxxdots = self.xdots + (xs - self.xs) / h
        relaxrod = ElasticRodS(
                xs=xs,
                restl=self.restl,
                xdots=relaxxdots,
                thetas=self.thetas,
                omegas=self.omegas,
                #xdots=None,
                #thetas=None,
                #omegas=None,
                #refd1s=None,
                #refd2s=None,
                rho=self.rho)
        relaxrod.InitTF(self)
        relaxrod.init_xs = self.xs
        relaxrod.init_op = tf.assign(relaxrod.xs, relaxrod.init_xs)
        relaxrod.loss = relaxrod.GetEConstraintTF()
        relaxrod.trainer = tf.train.AdamOptimizer(learning_rate)
        relaxrod.grads = relaxrod.trainer.compute_gradients(relaxrod.loss)
        # print('relaxrod.grads {}'.format(relaxrod.grads))
        # print('relaxrod.fullrestvl {}'.format(relaxrod.fullrestvl))
        # print('relaxrod.rho {}'.format(relaxrod.rho))
        relaxrod.weighted_grads = \
            [(grad[0] / _paddim(relaxrod.fullrestvl * relaxrod.rho), grad[1]) for grad in relaxrod.grads]
        relaxrod.apply_grads_op = relaxrod.trainer.apply_gradients(relaxrod.weighted_grads)
        if (not self.refd1s is None) and (not self.refd2s is None):
            TFPropogateRefDs(self, relaxrod, normalize=True)
        return relaxrod

    def GetVariableList(self):
        return [self.xs, self.xdots, self.thetas, self.omegas, self.refd1s, self.refd2s]

    def UpdateVariable(self, vl):
        [self.xs, self.xdots, self.thetas, self.omegas, self.refd1s, self.refd2s] = vl

    def Relax(self, sess, irod, icond, ccd_h=None, SelS=None, options=None, run_metadata=None):
        h = ccd_h
        inputdict = helper.create_dict([irod], [icond])
        sess.run(self.init_op, feed_dict=inputdict, options=options, run_metadata=run_metadata)
        while True:
            #init_xs = sess.run(self.init_xs, feed_dict=inputdict)
                #    orod.thetas, orod.omegas], feed_dict=inputdict)
            for i in range(100):
                # sess.run(self.train_op, feed_dict=inputdict)
                E, _ = sess.run([self.loss, self.apply_grads_op], feed_dict=inputdict, options=options, run_metadata=run_metadata)
                if i % 1 == 0:
                    # print('loss (iter:{}): {}'.format(i, E))
                    if math.fabs(E) < 1e-9:
                        # print('Leaving at Iter {}'.format(i))
                        break
                '''
                if i % 100 == 0:
                    E = sess.run(self.loss, feed_dict=inputdict)
                    print('loss: {}'.format(E))
                '''
            if h is not None:
                ccddict = helper.create_dict([irod], [icond])
                ccddict.update({self.sela: SelS[0], self.selb: SelS[1]})
                leaving = self.DetectAndApplyImpulse(sess, h, ccddict)
                if leaving:
                    break
            else:
                break
        vl = sess.run(self.GetVariableList(), feed_dict=inputdict, options=options, run_metadata=run_metadata)
        icond.UpdateVariable(vl)
        return icond

    '''
    Functions with 'TF' suffix assume ElasticRod object members are tensors.
    '''
    def InitTF(rod, other):
        '''
        Calculate Intermediate Tensors (e.g. Curvatures) from Input Placeholders
        This is mandantory for Energy Terms
        '''
        rod.evec = TFGetEdgeVector(rod.xs)
        rod.tans = _normalize(rod.evec)
        rod.fullrestvl = TFGetVoronoiEdgeLength(rod.restl)
        rod.innerrestvl = _trimslices(rod.fullrestvl, _lastdim(rod.fullrestvl), [1, 1])
        rod.ks = TFGetCurvature(rod.evec, rod.restl)
        return rod.clone_args_from(other)

    def GetEConstraintTF(rod):
        diff = TFGetLengthConstraintFunction(rod)
        total = tf.reduce_sum(_dot(diff, diff))
        if rod.anchors is not None:
            '''
            Pick up (:,0,:) or (0,:) from rod.xs
            as 2D or 1D tensor
            '''
            ndim = _ndim(rod.xs)
            start = list([0] * ndim)
            size = list([-1] * ndim)
            size[-2] = 1
            firstX = tf.unstack(tf.slice(rod.xs, start, size), axis=ndim - 2)[0]
            diff = firstX - rod.anchors
            total += tf.reduce_sum(_dot(diff, diff))
        return total

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
        # print('rod.ks {}'.format(rod.ks))
        # print('refd1primes {}'.format(refd1primes))
        rod.mbars = _dot(refd1primes, _trimslices(rod.refd2s, margins=[0,1]))
        # print('rod.mbars {}'.format(rod.mbars))
        # print('rod.thetas {}'.format(rod.thetas))
        theta_i_1, theta_i = _diffslices(rod.thetas, dim=_lastdim(rod.thetas))
        # print('theta_i {}'.format(theta_i))
        difftheta = theta_i - theta_i_1
        # print('difftheta {}'.format(difftheta))
        deltatheta = difftheta - rod.mbars
        # print('deltatheta {}'.format(deltatheta))
        return tf.reduce_sum(tf.multiply(deltatheta*deltatheta, 1.0/rod.innerrestvl))

    def GetEGravityTF(rod):
        z_begin = list([0] * len(rod.xs.get_shape()))
        z_begin[-1] = 2
        z_size = list([-1] * len(rod.xs.get_shape()))
        Zs = tf.slice(rod.xs, z_begin, z_size) - rod.floor_z
        return 0.5 * rod.g * tf.reduce_sum(tf.multiply(Zs, Zs) * _paddim(rod.fullrestvl))

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

    # TODO: Node to Calculate sela/selb
    def CreateCCDNode(self, irod, h):
        self.sela = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.selb = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.impulse_with_sels = TFRodCollisionImpulse(h, irod, self, self, self.sela, self.selb)
        self.appled_xs = TFApplyImpulse(h, self,\
                self.impulse_with_sels[1],\
                self.impulse_with_sels[2],\
                self.impulse_with_sels[0]
                )
        self.apply_impulse_op = tf.assign(self.xs, self.appled_xs)
        # print(self.apply_impulse_op)
        return self

    def DetectAndApplyImpulse(self, sess, h, inputdict):
        # print('xs before impulse {}'.format(sess.run(self.xs, feed_dict=inputdict)))
        sel,_ = sess.run([self.impulse_with_sels[1], self.apply_impulse_op], feed_dict=inputdict)
        # print(sess.run(self.impulse_with_sels, feed_dict=inputdict))
        # print('xs after impulse {}'.format(sess.run(self.xs, feed_dict=inputdict)))
        return len(sel) == 0
        # return True

def TFInitRod(rod):
    return rod.InitTF(None)

def TFGetEConstraint(rod):
    return rod.GetEConstraintTF()

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

