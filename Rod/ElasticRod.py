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

_epsilon = 1e-5
_default_rho = 0.1
_stiff = 1e7

'''
Helper functions to handle tensors

TODO: move into another file
'''

def _ndim(tensor):
    return len(tensor.get_shape())

def _lastdim(tensor):
    return _ndim(tensor) - 1

def _dot(tensor1, tensor2, dim=None, keep_dims=False):
    if dim == None:
        dim = _lastdim(tensor1)
    return tf.reduce_sum(tf.multiply(tensor1, tensor2), dim, keep_dims=keep_dims)

def _sqnorm(tensor, dim=None, keep_dims=False):
    return _dot(tensor, tensor, dim, keep_dims)

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
    return evec/norm

def _paddim(tensor):
    return tf.stack([tensor], axis=_lastdim(tensor)+1)

def _pick_segment_from_rods(tensor, segid):
    ndim = _ndim(tensor)
    assert ndim == 3
    start = list([0] * ndim)
    start[1] = segid
    size = list([-1] * ndim)
    size[1] = 1
    rodsseg = tf.slice(tensor, start, size)
    dshape = tf.shape(rodsseg)
    return tf.reshape(rodsseg, shape=[dshape[0], dshape[2]])

def _sinh(tensor):
    return (tf.exp(tensor) - tf.exp(-tensor)) / 2.0

def _asinh(tensor):
    return tf.log(tensor + tf.sqrt(tensor * tensor + 1))

'''
Helper functions for elastic rods
'''

def TFGetEdgeVector(xs):
    x_i_1, x_i = _diffslices(xs)
    return x_i - x_i_1

def TFGetEdgeLength(evec):
    norms = tf.norm(evec, axis=_lastdim(evec))
    return tf.reshape(norms, [evec.get_shape().as_list()[0]])

def TFGetVoronoiEdgeLength(enorms):
    en_i_1, en_i = _diffslices(enorms)
    paddings = [[0,0] for i in range(len(enorms.get_shape()))]
    paddings[-1] = [0,1]
    en_i_1 = tf.pad(enorms, paddings, 'CONSTANT')
    paddings[-1] = [1,0]
    en_i   = tf.pad(enorms, paddings, 'CONSTANT')
    return (en_i_1 + en_i) / 2

def TFGetCurvature(ev, enorms):
    e_i_1, e_i = _diffslices(ev)
    en_i_1, en_i = _diffslices(enorms, dim=_lastdim(enorms))
    denominator1 = en_i_1 * en_i
    denominator2 = _dot(e_i_1, e_i)
    denominator = (denominator1+denominator2)
    shape3 = denominator.get_shape().as_list()
    denominator = _paddim(denominator)
    return 2 * tf.multiply(tf.cross(e_i_1, e_i), 1.0/(denominator))

def TFGetLengthConstraintFunction(rod):
    sqlen = _dot(rod.evec, rod.evec)
    sqrest = rod.restl * rod.restl;
    return sqlen - sqrest

# For unit \alpha
def TFParallelTransportQuaternion(prod, crod):
    axes = -tf.cross(prod.tans, crod.tans)
    cosines = _dot(prod.tans, crod.tans)
    halfconsines = _paddim(tf.sqrt((cosines + 1)/2.0))
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
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    row1 = tf.concat([aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], axis=lastdim)
    row2 = tf.concat([2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], axis=lastdim)
    row3 = tf.concat([2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc], axis=lastdim)
    R = tf.stack([row1, row2, row3], axis=lastdim)
    shape = prod.refd1s.get_shape()
    refd1s = _paddim(prod.refd1s)
    refd2s = _paddim(prod.refd2s)
    crod.refd1s = tf.reshape(tf.matmul(R, refd1s), shape)
    crod.refd2s = tf.reshape(tf.matmul(R, refd2s), shape)
    if normalize:
        crod.refd1s = _normalize(crod.refd1s)
        crod.refd2s = _normalize(crod.refd2s)
    return crod

def TFRodXSel(xs, Sel):
    xs_i_1, xs_i = _diffslices(xs, dim=_lastdim(xs)-1)
    gxs_i_1 = tf.gather_nd(xs_i_1, Sel)
    gxs_i = tf.gather_nd(xs_i, Sel)
    return gxs_i_1, gxs_i

def TFRodXDotSel(xdots, Sel):
    xdots_i_1, xdots_i = _diffslices(xdots, dim=_lastdim(xdots)-1)
    gxdots_i_1 = tf.gather_nd(xdots_i_1, Sel)
    gxdots_i = tf.gather_nd(xdots_i, Sel)
    return (gxdots_i_1 + gxdots_i) / 2.0

def TFSignedVolumes(xs, SelA, SelB):
    gaxs_i_1, gaxs_i = TFRodXSel(xs, SelA)
    gbxs_i_1, gbxs_i = TFRodXSel(xs, SelB)
    avec = gaxs_i_1 - gbxs_i_1
    bvec = gaxs_i - gbxs_i_1
    cvec = gbxs_i - gbxs_i_1
    return _dot(avec, tf.cross(bvec, cvec))


'''
Narrow phase of collision detection
'''

def _tri(c, a, b):
    '''
    Returns c . (a x b)
    '''
    return _dot(c, tf.cross(a,b), keep_dims=True)

def _roots_in_range(roots, minimal=0.0, maximal=1.0):
    return tf.logical_and(tf.less_equal(roots, maximal), tf.greater_equal(roots, minimal))

def _clamp_roots(roots):
    valids = tf.logical_and(tf.less_equal(roots, 1.0), tf.greater_equal(roots, 0.0))
    fpbool = tf.to_float(valids)
    return roots * fpbool - (1.0 - fpbool) * 10.0

def TFRodCCD_Select(crod, nrod, SelS):
    Q_sv, Q_ev = TFRodXSel(crod.xs, SelS)
    nQ_sv, nQ_ev = TFRodXSel(nrod.xs, SelS)
    Qdot_sv = nQ_sv - Q_sv
    Qdot_ev = nQ_ev - Q_ev
    return tf.to_double(Q_sv), tf.to_double(Q_ev), tf.to_double(Qdot_sv), tf.to_double(Qdot_ev)

def TFRod_RealCCD(crod, nrod, srod, ASelS, BSelS):
    '''
    TFRod_RealCCD: solve the continuous collision detection formula
    Returns a tensor of booleans indicating which pairs in ASelS and BSelS are colliding

    Q_a_sv, Q_a_ev: A rod Starting Vertex and Ending Vertex
    Qdot_a_sv, Qdot_a_ev: Displacement b/w Q and nQ
    '''
    Q_a_sv, Q_a_ev, Qdot_a_sv, Qdot_a_ev = TFRodCCD_Select(crod, nrod, ASelS)
    Q_b_sv, Q_b_ev, Qdot_b_sv, Qdot_b_ev = TFRodCCD_Select(crod, nrod, BSelS)
    p0 = Q_a_ev - Q_a_sv
    v0 = Qdot_a_ev - Qdot_a_sv
    p1 = Q_b_ev - Q_a_sv
    v1 = Qdot_b_ev - Qdot_a_sv
    p2 = Q_b_sv - Q_a_sv
    v2 = Qdot_b_sv - Qdot_a_sv
    '''
    a x^3 + b x^2 + c x + d = 0
    '''
    d = _tri(p2, p0, p1)
    c = _tri(p2, p0, v1) + _tri(p2, v0, p1) + _tri(v2, p0, p1)
    b = _tri(p2, v0, v1) + _tri(v2, p0, v1) + _tri(v2, v0, p1)
    a = _tri(v2, v0, v1)

    def _within1_from_tau(a, b, c, dbg=None):
        '''
        Return whether (s, t) are in [0,1]
        '''
        axb = tf.cross(a,b)
        sqn = _sqnorm(axb, keep_dims=True)
        s = _dot(tf.cross(c,b), axb, keep_dims=True) / sqn
        sgood = _roots_in_range(s)
        t = _dot(tf.cross(c,a), axb, keep_dims=True) / sqn
        tgood = _roots_in_range(t)
        if dbg is not None:
            dbg.dbg_s = s
            dbg.dbg_t = t
            dbg.dbg_sgood = sgood
            dbg.dbg_tgood = tgood
        return tf.logical_and(sgood, tgood)

    def _valid_from_tau(taus, dbg=None):
        '''
        Return whether tau and the corresponding (s, t) are in [0,1]
        '''
        p = Q_a_sv + taus * Qdot_a_sv
        r = Q_a_ev + taus * Qdot_a_ev - p
        q = Q_b_sv + taus * Qdot_b_sv
        s = Q_b_ev + taus * Qdot_b_ev - q
        t = q - p
        rxs = tf.cross(r, s)
        nz = tf.greater(_sqnorm(rxs, keep_dims=True), 0.0)
        taugood = _roots_in_range(taus)
        nz = tf.logical_and(taugood, nz)
        if dbg is not None:
            dbg.dbg_taus = taus
            dbg.dbg_taugood = taugood
            dbg.dbg_rxs = rxs
            dbg.dbg_rxsnz = nz
            dbg.dbg_q_pxr = tf.cross(q - p, r)
        return tf.logical_and(nz, tf.where(nz, _within1_from_tau(r, s, t, dbg), nz))

    def _valid_quadratic_roots(a,b,c, dbg=None):
        '''
        Detect collision given a x^2 + b x + c
        '''
        det = b*b - 4 * a * c
        def _first_valid_det_ge_0(a,b,c,det):
            sqrtdet = tf.sqrt(det)
            roots1 = (-b+sqrtdet)/(2*a)
            roots2 = (-b-sqrtdet)/(2*a)
            if dbg is not None:
                dbg.dbg_quadroots1 = roots1
                dbg.dbg_quadroots2 = roots2
            return tf.logical_or(_valid_from_tau(roots1), _valid_from_tau(roots2, dbg))
        return tf.where(tf.greater_equal(det, 0.0),
                _first_valid_det_ge_0(a,b,c,det),
                tf.greater_equal(det, 0.0))

    def _valid_linear_roots(a,b, dbg=None):
        vanished = tf.less_equal(tf.abs(a), _epsilon)
        linvalid = _valid_from_tau(-b/a)
        if dbg is not None:
            dbg.dbg_linroot = -b/a
            dbg.dbg_linvalid = linvalid
        return tf.where(vanished, tf.equal(b, 0.0), linvalid)

    def _valid_quadratic_or_linear_roots(a,b,c, dbg=None):
        return tf.where(tf.equal(tf.abs(a), 0.0), _valid_linear_roots(b,c, dbg), _valid_quadratic_roots(a,b,c, dbg))

    def _valid_cubic_tri_roots(p, q, a, b, dbg=None):
        A = 2 * tf.sqrt(-p/3)
        Phi = tf.acos(3*q/(A*p))
        B = - b/(3*a)
        roots1 = A*tf.cos(Phi/3)+B
        roots2 = A*tf.cos((Phi + 2.0 * math.pi)/3)+B
        roots3 = A*tf.cos((Phi + 4.0 * math.pi)/3)+B
        if dbg is not None:
            dbg.dbg_A = A
            dbg.dbg_Phi = Phi
            dbg.dbg_B = B
            dbg.dbg_triroots1 = roots1
            dbg.dbg_triroots2 = roots2
            dbg.dbg_triroots3 = roots3
        return tf.logical_or(tf.logical_or(_valid_from_tau(roots1), _valid_from_tau(roots2)), _valid_from_tau(roots3))

    def _valid_cubic_single_root(p, q, a, b, dbg=None):
        Abar = 2 * tf.sqrt(p/3)
        Phibar = _asinh(3 * q / (Abar * p))
        roots1 = -2*tf.sqrt(p/3)*_sinh(Phibar/3.0) - b/(3*a)
        if dbg is not None:
            dbg.dbg_signleroots = roots1
            dbg.dbg_Abar = Abar
            dbg.dbg_Phibar = Phibar
        return _valid_from_tau(roots1)

    def _valid_cubic_roots(a, b, c, d, dbg=None):
        p = (3 * a * c - b * b)/(3 * a * a)
        q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d)/(27 * a * a * a)
        if dbg is not None:
            dbg.dbg_p = p
            dbg.dbg_q = q
        return tf.where(tf.less_equal(p, 0.0),
                _valid_cubic_tri_roots(p, q, a, b, dbg),
                _valid_cubic_single_root(p, q, a, b, dbg))

    cancoltv = tf.where(tf.abs(a) > _epsilon,\
            _valid_cubic_roots(a,b,c,d,srod),\
            _valid_quadratic_or_linear_roots(b,c,d,srod))
    srod.dbg_a = a
    srod.dbg_b = b
    srod.dbg_c = c
    srod.dbg_d = d
    return tf.reshape(cancoltv, [tf.shape(cancoltv)[0]])

def TFRodCCDExtended_signed_volume_based(crod, nrod, srod, ASelS, BSelS):
    cvolumes = TFSignedVolumes(crod.xs, ASelS, BSelS)
    nvolumes = TFSignedVolumes(nrod.xs, ASelS, BSelS)
    return tf.not_equal(tf.sign(cvolumes), tf.sign(nvolumes))

def TFRodCCDExtended(crod, nrod, srod, ASelS, BSelS):
    return TFRod_RealCCD(crod, nrod, srod, ASelS, BSelS)

def TFRodCCD(crod, nrod, srod, ASelS, BSelS):
    a_list = TFRodCCDExtended(crod, nrod, srod, ASelS, BSelS)
    return a_list[0]

def CollisionFilter(SelS_in, collision):
    return tf.gather_nd(SelS_in, tf.where(tf.equal(collision, True)))

'''
Collision Handling
'''

def TFRodCollisionImpulse(h, crod, nrod, srod, ASelS_in, BSelS_in, collision = None):
    gcxs_k = None
    if collision is None:
        collision = TFRodCCDExtended(crod, nrod, srod, ASelS_in, BSelS_in)
    ASelS = CollisionFilter(ASelS_in, collision)
    BSelS = CollisionFilter(BSelS_in, collision)

    ''' Gathered Current Xs for A '''
    gcxs_k_1, gcxs_k = TFRodXSel(crod.xs, ASelS)
    ''' Gathered Next Xs for A '''
    gnxs_k_1, gnxs_k = TFRodXSel(nrod.xs, ASelS)

    gqdots = ((gnxs_k_1 - gcxs_k_1) + (gnxs_k - gcxs_k)) / 2.0

    ''' Gathered Current Xs for B '''
    gcxs_k_1_b, gcxs_k_b = TFRodXSel(crod.xs, BSelS)
    ''' Gathered Next Xs for B '''
    gnxs_k_1_b, gnxs_k_b = TFRodXSel(nrod.xs, BSelS)
    grqdots = ((gnxs_k_1_b - gcxs_k_1_b) + (gnxs_k_b - gcxs_k_b)) / 2.0
    relqdots = gqdots - grqdots
    gtans = tf.gather_nd(srod.tans, BSelS)

    gnormals = tf.cross(gtans, tf.cross(gtans, relqdots))
    gsegmass = _paddim(tf.gather_nd(nrod.restl * nrod.rho, ASelS))
    return gnormals * gsegmass, ASelS, BSelS

def TFApplyImpulse(h, rods, ASelS, BSelS, impulse, factor):
    '''
    Return the xs with impulse applied
    (Note we are not taking the penalty impulse method strictly)
    '''
    ASelX = ASelS
    BSelX = BSelS
    ASelX_p1 = ASelS + tf.constant([0,1])
    BSelX_p1 = BSelS + tf.constant([0,1])
    indices = tf.concat([ASelS, ASelX_p1 , BSelS, BSelX_p1], axis=0)
    gAvertmass = _paddim(tf.gather_nd(rods.fullrestvl * rods.rho, ASelX))
    gAvertmass_p1 = _paddim(tf.gather_nd(rods.fullrestvl * rods.rho, ASelX_p1))
    gBvertmass = _paddim(tf.gather_nd(rods.fullrestvl * rods.rho, BSelX))
    gBvertmass_p1 = _paddim(tf.gather_nd(rods.fullrestvl * rods.rho, BSelX_p1))
    # factor = 2.0
    impulse_to_A = factor * impulse / gAvertmass
    impulse_to_A_p1 = factor * impulse / gAvertmass_p1
    impulse_to_B = -factor * impulse / gBvertmass
    impulse_to_B_p1 = -factor * impulse / gBvertmass_p1
    updates = tf.concat([impulse_to_A, impulse_to_A_p1, impulse_to_B, impulse_to_B_p1], axis=0)
    return tf.scatter_nd_add(rods.xs, indices, updates)

'''
Broadphase for collision detection
'''

def _translate_from_stacked(nrods, segidi, segidj, rodids):
    cond = tf.less(rodids, nrods)
    unit_col = tf.ones_like(rodids, dtype=tf.int64)
    icol = unit_col * tf.convert_to_tensor(segidi, dtype=tf.int64)
    jcol = unit_col * tf.convert_to_tensor(segidj, dtype=tf.int64)
    return tf.where(cond, tf.stack([rodids, icol], axis=1), tf.stack([rodids - nrods, jcol], axis=1))

# Innerloop (#, i) vs (#, j) for i == j
def TFSegmentVsSegmentDistanceFilterNoCat(h, i, j, xs, sqnxs, thresh, to_stack):
    nrods = tf.cast(tf.shape(xs)[0], tf.int64)
    xst = _pick_segment_from_rods(xs, i)
    crossterm = -2 * tf.matmul(xst, xst, transpose_b=True)
    sqnxst = _pick_segment_from_rods(sqnxs, i)
    D = (crossterm + sqnxst) + tf.transpose(sqnxst)
    indices = tf.where(tf.less(D, thresh * h))
    Arods,Brods = tf.unstack(indices, 2, axis=1)
    processed = tf.where(tf.less(Arods, Brods)) # Only consider pairs (Alpha, i) (Beta, j) where Alpha < Beta
    ArodsRemain = tf.gather_nd(Arods, processed)
    BrodsRemain = tf.gather_nd(Brods, processed)
    ArodsRelabel = _translate_from_stacked(nrods, i, j, ArodsRemain)
    BrodsRelabel = _translate_from_stacked(nrods, i, j, BrodsRemain)
    tmp1 = tf.concat([ArodsRelabel, BrodsRelabel], axis=_lastdim(ArodsRelabel))
    tmp2 = tf.concat([to_stack, tmp1], axis=0)
    return tmp2

# Innerloop (#, i) vs (#, j) for i != j
def TFSegmentVsSegmentDistanceFilterNoInnerCross(h, i, j, xs, sqnxs, thresh, to_stack):
    nrods = tf.cast(tf.shape(xs)[0], tf.int64)
    xsi = _pick_segment_from_rods(xs, i)
    xsj = _pick_segment_from_rods(xs, j)
    crossterm = -2 * tf.matmul(xsi, xsj, transpose_b=True)
    sqnxsi = _pick_segment_from_rods(sqnxs, i)
    sqnxsj = _pick_segment_from_rods(sqnxs, j)
    D = (crossterm + sqnxsi) + tf.transpose(sqnxsj)
    indices = tf.where(tf.less(D, thresh * h))
    Arods,Brods = tf.unstack(indices, 2, axis=1)
    '''
    Disable Self-Colliding Detection
    Only detect collision b/w different rods
    '''
    processed = tf.where(tf.not_equal(Arods, Brods))
    ArodsRemain = tf.gather_nd(Arods, processed)
    BrodsRemain = tf.gather_nd(Brods, processed)
    unit_col = tf.ones_like(ArodsRemain, dtype=tf.int64)
    icol = unit_col * tf.convert_to_tensor(i, dtype=tf.int64)
    jcol = unit_col * tf.convert_to_tensor(j, dtype=tf.int64)
    ArodsRelabel = tf.stack([ArodsRemain, icol], axis=1)
    BrodsRelabel = tf.stack([BrodsRemain, jcol], axis=1)
    tmp1 = tf.concat([ArodsRelabel, BrodsRelabel], axis=_lastdim(ArodsRelabel))
    tmp2 = tf.concat([to_stack, tmp1], axis=0)
    return tmp2

def TFSpecificSegmentDistanceFilter(h, maxsegidx, midpoints, sqmidpoints, segmaxvel, to_stack):
    '''
    Middle loop of TFDistanceFilter
    Generate selectors for close-enough midpoint pairs, given one segment
    '''
    # print(type(idxseg))
    j0 = tf.constant(0, dtype=tf.int64)
    loop_cond = lambda j, _1, _2, _3, _4: tf.less(j, maxsegidx)
    loop_body = lambda j, xs, sqnxs, xdots, to_stack: \
        [tf.add(j, 1), xs, sqnxs, xdots, \
         TFSegmentVsSegmentDistanceFilterNoInnerCross(h, j, maxsegidx, xs, sqnxs, xdots, to_stack)]
    ivalues = [j0, midpoints, sqmidpoints, segmaxvel, to_stack]
    shape_inv = list(map(lambda x:x.get_shape(), ivalues))
    shape_inv[-1] = tf.TensorShape([None,4])
    loop = tf.while_loop(loop_cond, loop_body, loop_vars=ivalues, shape_invariants=shape_inv)
    return TFSegmentVsSegmentDistanceFilterNoCat(h, maxsegidx, maxsegidx, midpoints, sqmidpoints, segmaxvel, loop[-1])

def TFDistanceFilter(h, midpoints, thresholds):
    '''
    Outer loop of TFDistanceFilter
    Generate selectors for all close-enough midpoint pairs
    '''
    sqnmidpoints = _dot(midpoints, midpoints, keep_dims=True)
    nsegs = tf.cast(tf.shape(midpoints)[1], dtype=tf.int64)
    i0 = tf.constant(0, dtype=tf.int64)
    stack0 = -1 * tf.ones(shape=[1,4], dtype=tf.int64)
    loop_cond = lambda i, _1, _2, _3, _4: tf.less(i, nsegs)
    loop_body = lambda i, xs, sqnxs, xdots, stack: [tf.add(i, 1), xs, sqnxs, xdots, TFSpecificSegmentDistanceFilter(h, i, xs, sqnxs, xdots, stack)]
    ivalues = [i0, midpoints, sqnmidpoints, thresholds, stack0]
    shape_inv = list(map(lambda x:x.get_shape(), ivalues))
    shape_inv[-1] = tf.TensorShape([None,4])
    loop = tf.while_loop(loop_cond, loop_body, loop_vars=ivalues, shape_invariants=shape_inv)
    return tf.slice(loop[-1], [1, 0], [-1, -1])

'''
ElasticRodS class:
    All elastic rods and segments within them
'''

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

    xs = None # vertex positions
    thetas = None # segment rotations w.r.t. reference frame
    restl = None # Rest length
    omegas = None
    midpoints = None # Cached

    tans = None
    refd1s = None
    refd2s = None

    rho = _default_rho
    alpha = 1.0 # Bending stiffness
    beta = 1.0 # Twisting stiffness
    g = 0.0 # Gravity
    floor_z = -50.0 # Floor Z
    constraint_tolerance = 1e-9
    constraint_iterations = 100
    constraint_learning_rate = 1e-4
    anchor_stiffness = 1
    obstacle_impulse_op = None

    ccd_threshold = None
    ccd_factor = None
    CH_factor = 1.0

    def clone_args_from(self, other):
        '''
        Copying parameters from another object
        '''
        if other is None:
            return self
        self.rho = other.rho
        self.alpha = other.alpha
        self.beta = other.beta
        self.g = other.g
        self.floor_z = other.floor_z
        self.constraint_tolerance = other.constraint_tolerance
        self.constraint_iterations = other.constraint_iterations
        self.constraint_learning_rate = other.constraint_learning_rate
        self.anchor_stiffness = other.anchor_stiffness
        self.anchors = other.anchors
        self.sparse_anchor_indices = other.sparse_anchor_indices
        self.sparse_anchor_values = other.sparse_anchor_values
        self.obstacle_impulse_op = other.obstacle_impulse_op
        self.CH_factor = other.CH_factor
        return self

    c = None # Translation
    q = None # Rotation quaternion
    anchors = None # 2D tensor: N Rod x 3, representing root of each rod
    sparse_anchor_indices = None # 2D tensor, [None] x 2
    sparse_anchor_values = None # 2D tensor, [None] x 3
    sela = None
    selb = None

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

        '''
        Return the ElasticRodS object for the next time frame
        -- without applying constraints and handling collisions
        '''

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
                + pseudonrod.GetEGravityTF()
        XForce, TForce = tf.gradients(-E, [pseudonrod.xs, pseudonrod.thetas])
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

    def CalcPenaltyRelaxationTF(self, h, learning_rate=None):

        '''
        Return a ElasticRodS object for current time frame
        with constraints and collisions

        Call this function on the object returned by CalcNextRod.
        '''

        if learning_rate is None:
            learning_rate = self.constraint_learning_rate

        xs = tf.Variable(np.zeros(shape=self.xs.get_shape().as_list(), dtype=np.float32), name='xs')
        relaxxdots = self.xdots + (xs - self.xs) / h
        relaxrod = ElasticRodS(
                xs=xs,
                restl=self.restl,
                xdots=relaxxdots,
                thetas=self.thetas,
                omegas=self.omegas,
                rho=self.rho)
        relaxrod.InitTF(self)
        relaxrod.init_xs = self.xs
        relaxrod.init_op = tf.assign(relaxrod.xs, relaxrod.init_xs)
        relaxrod.loss = relaxrod.GetEConstraintTF()
        relaxrod.trainer = tf.train.AdamOptimizer(learning_rate)
        relaxrod.grads = relaxrod.trainer.compute_gradients(relaxrod.loss)
        '''
        Divide the force by the mass to get acceleration
        '''
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

    def Relax(self, sess, irod, icond, ccd_h=None, ccd_broadthresh=10.0, options=None, run_metadata=None):
        '''
        Relaxation Step

        For constraints and collision handling
        '''
        h = ccd_h
        inputdict = helper.create_dict([irod], [icond])
        sess.run(self.init_op, feed_dict=inputdict, options=options, run_metadata=run_metadata)
        CH = self.CH_factor
        for C in range(100):
            leaving_iter = 0
            E = 0.0
            ''' Enforcing constratins by optimizing the penalty energy. '''
            for i in range(self.constraint_iterations):
                E, _ = sess.run([self.loss, self.apply_grads_op], feed_dict=inputdict, options=options, run_metadata=run_metadata)
                leaving_iter = i
                if i % 1 == 0:
                    # print('loss (iter:{}): {}'.format(i, E))
                    if math.fabs(E) < self.constraint_tolerance:
                        # print('Leaving at Iter {}'.format(i))
                        break
            if math.fabs(E) > 1e3:
                    print('Exploded with loss: {}'.format(E))
            if self.obstacle_impulse_op is not None:
                obstacle_impulse = sess.run(self.obstacle_impulse_op, feed_dict=inputdict, options=options, run_metadata=run_metadata)
            ''' Collision handling '''
            if h is not None:
                ccddict = helper.create_dict([irod], [icond])
                ccddict.update({self.ccd_threshold: ccd_broadthresh, self.ccd_factor:CH})
                leaving = self.DetectAndApplyImpulse(sess, h, ccddict)
                if leaving:
                    break
                AdaptiveCH = CH * 0.1
                AccumulatedCH = CH
                while not self.DetectAndApplyImpulse(sess, h, ccddict):
                    AccumulatedCH += AdaptiveCH
                    # AdaptiveCH *= 1.1
                    AdaptiveCH += CH * 0.1
                    ccddict.update({self.ccd_factor:AdaptiveCH})
                    pass
            else:
                break
        vl = sess.run(self.GetVariableList(), feed_dict=inputdict, options=options, run_metadata=run_metadata)
        icond.UpdateVariable(vl)
        return icond

    '''
    Note: Functions with 'TF' suffix assume ElasticRod object members are tensors.
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
        '''
        Penalty Potential
        '''
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
            size[-2] = 2
            firstX = tf.unstack(tf.slice(rod.xs, start, size), axis=ndim - 2)[0]
            diff = firstX - rod.anchors
            total += tf.reduce_sum(_dot(diff, diff)) * rod.anchor_stiffness
        elif rod.sparse_anchor_indices is not None:
            gxs = tf.gather_nd(rod.xs, rod.sparse_anchor_indices)
            diff = rod.sparse_anchor_values - gxs
            total += tf.reduce_sum(_dot(diff, diff)) * rod.anchor_stiffness
        return total

    def GetEBendTF(rod):
        '''
        Bending energy for unit alpha
        '''
        sqnorm = _dot(rod.ks, rod.ks)
        return tf.reduce_sum(tf.multiply(sqnorm, 1.0/rod.innerrestvl))

    def GetETwistTF(rod):
        '''
        For unit beta
        This should be a function w.r.t. thetas and xs
        which means difftheta also depends on xs
        '''
        refd1primes = tf.cross(rod.ks, _trimslices(rod.refd1s, margins=[0,1]))
        rod.mbars = _dot(refd1primes, _trimslices(rod.refd2s, margins=[0,1]))
        theta_i_1, theta_i = _diffslices(rod.thetas, dim=_lastdim(rod.thetas))
        difftheta = theta_i - theta_i_1
        deltatheta = difftheta - rod.mbars
        return tf.reduce_sum(tf.multiply(deltatheta*deltatheta, 1.0/rod.innerrestvl))

    def GetEGravityTF(rod):
        '''
        Gravity potential
        '''
        if rod.g == 0.0:
            return tf.constant(0.0)
        z_begin = list([0] * len(rod.xs.get_shape()))
        z_begin[-1] = 2
        z_size = list([-1] * len(rod.xs.get_shape()))
        Zs = tf.slice(rod.xs, z_begin, z_size) - rod.floor_z
        return 0.5 * rod.g * tf.reduce_sum(tf.multiply(Zs, Zs) * _paddim(rod.fullrestvl * rod.rho))

    def TFKineticI(rodnow, rodnext, h):
        '''
        Kinetic Energy from positions
        Legacy API for Lagrangian
        '''
        rodnow.xdots = 1/h * (rodnext.xs - rodnow.xs)
        return TFKineticD(rodnow)

    def TFKineticD(rod):
        '''
        Kinetic Energy Directly from velocity, For unit \rho
        Legacy API for Lagrangian
        '''
        xdot = rod.xdots
        xdot_i_1, xdot_i = _diffslices(xdot, 0)
        avexdot = 0.5 * (xdot_i_1 + xdot_i)
        sqnorm = tf.reduce_sum(tf.multiply(avexdot, avexdot), 1, keep_dims=False)
        return 0.5 * tf.reduce_sum(rod.restl * rod.rho * sqnorm)

    def CreateCCDNode(self, irod, h, thresholds=None):
        '''
        Create sufficient computational nodes to detect and handle collisions
        '''
        midpoints = self.GetMidPointsTF()
        if thresholds is None:
            self.ccd_threshold = tf.placeholder(dtype=tf.float32)
        else:
            self.ccd_threshold = thresholds
        if self.sela is None:
            distance_pairs = TFDistanceFilter(h, midpoints, self.ccd_threshold)
            distance_pairs = tf.cast(distance_pairs, tf.int32)
            self.sela = tf.slice(distance_pairs, [0, 0], [-1, 2])
            self.selb = tf.slice(distance_pairs, [0, 2], [-1, 2])
        self.impulse_with_sels = TFRodCollisionImpulse(h, irod, self, self, self.sela, self.selb)
        self.ccd_factor = tf.placeholder(dtype=tf.float32)
        self.applied_xs = TFApplyImpulse(h, self,\
                self.impulse_with_sels[1],\
                self.impulse_with_sels[2],\
                self.impulse_with_sels[0],
                self.ccd_factor
                )
        self.apply_impulse_op = tf.assign(self.xs, self.applied_xs)
        return self

    def DetectAndApplyImpulse(self, sess, h, inputdict):
        '''
        Handle the collision in a TF session
        '''
        ops = [tf.concat([self.impulse_with_sels[1], self.impulse_with_sels[2]], axis=1),\
                self.impulse_with_sels[0],\
                self.apply_impulse_op]
        sels_results, impulse_results, _ = sess.run(ops, feed_dict=inputdict)
        return len(sels_results) == 0

    def GetMidPointsTF(self):
        if self.midpoints is None:
            xs_i_1, xs_i = _diffslices(self.xs)
            self.midpoints = (xs_i_1 + xs_i) / 2.0
        return self.midpoints

'''
Legacy APIs for test_V, test_T and test_PT
'''

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
    return 0.5 * tf.reduce_sum(rod.restl * rod.rho * sqnorm)
