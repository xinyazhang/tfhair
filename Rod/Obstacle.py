import tensorflow as tf
import numpy as np
import RodHelper as helper
import math

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

class SphericalBodyS(object):

    """
    This is a class that models obstacles approximated by spheres.
    Constructor takes a list of spheres, i.e. center, radius (possibly velocity?).
    Center, and radius, of course could be tensors, placeholders, etc
    """
    CoR = 0.1
    margin = 0.1       # spacing for collision

    def __init__(self, centers, radii):
        super(SphericalBodyS, self).__init__()
        self.centers = centers
        self.radii = radii

    def DetectAndApplyImpulseOp(self, h, rod):
        impulse_ops = []
        for center, radius in zip(self.centers, self.radii):
            impulse_op = self._DetectAndApplyImpulseEachOp(h, center, radius, rod)
            impulse_ops.append(impulse_op)
        return impulse_ops

    def _DetectAndApplyImpulseEachOp(self, h, center, radius, rod):
        # pts = rod.GetMidPointsTF()
        pts = rod.xs
        # check if already collided
        dvec = center -pts
        dist2 = tf.square(tf.norm(dvec, axis=-1))
        diff2 = dist2 - radius * radius
        # check if approaching
        # xdots_i_1, xdots_i = _diffslices(rod.xdots, dim=-2)
        # qdots = (xdots_i_1 + xdots_i) / 2.0
        qdots = rod.xdots
        relqdots = qdots    # FIXME: for now assume that obstacles are stationary
        approaching = tf.reduce_sum(relqdots * dvec, -1)
        # select both collided and approaching segments
        sel = tf.where(tf.logical_and(diff2 < self.margin, approaching > 0))
        relqdotsSel = tf.gather_nd(relqdots, sel)
        # compute impulse with coefficient of restitution
        factor = -(1 + self.CoR)
        impulse = factor * h * relqdotsSel
        applied_xs = tf.scatter_nd_add(rod.xs, sel, impulse)
        return tf.assign(rod.xs, applied_xs)
