import sys
import math
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
    Elastic Rod Class

    cpos: [n, 3] tensor
    cvel: [n, 3] tensor
    theta: [n, 1] tensor
    omega: [n, 3] tensor
    '''

    density = 1e3
    radius = 0.01
    penalty_stiffness = 10

    def __init__(self, cpos, cvel, theta, omega, restl):
        # primary variables for phase update
        self.cpos = cpos
        self.cvel = cvel
        self.theta = theta
        self.omega = omega
        self.restl = restl
        # next rod state
        self.nrod = None
        # init
        self._init()

    def _init(self):
        # constraints
        self.clength = None # length constraints
        # auxiliary variables to help calculation
        self.evec = self._compute_edge_vector(self.cpos)
        self.enorms = self._compute_edge_norm(self.evec)
        self.tan = self._compute_tangent(self.evec, self.enorms)
        self.kappa = self._compute_curvature(self.evec, self.enorms)
        self.fullrestvl = self._compute_full_restvl(self.restl)
        self.innerrestvl = self._compute_inner_restvl(self.fullrestvl)
        self.mass = self._compute_mass_matrix(self.radius, self.density, self.fullrestvl)

    def _compute_edge_vector(self, cpos):
        return cpos[1:,:] - cpos[0:-1,:]

    def _compute_edge_norm(self, evec):
        return tf.norm(evec, axis=1)

    def _compute_full_restvl(self, restl):
        en_i = tf.pad(restl, [[1, 0]], "CONSTANT")
        en_i_1 = tf.pad(restl, [[0, 1]], "CONSTANT")
        return (en_i + en_i_1) / 2.0

    def _compute_inner_restvl(self, fullrestvl):
        return fullrestvl[1:-1]

    def _compute_curvature(self, evec, enorms):
        e_i_1, e_i = evec[:-1], evec[1:]
        en_i_1, en_i = enorms[:-1], enorms[1:]
        denominator1 = en_i_1 * en_i
        denominator2 = tf.reduce_sum(tf.multiply(e_i_1, e_i), 1, keep_dims=False)
        denominator = (denominator1+denominator2)
        shape3 = denominator.get_shape().as_list()
        denominator = tf.reshape(denominator, [shape3[0],1])
        return 2 * tf.multiply(tf.cross(e_i_1, e_i), 1.0/(denominator))

    def _compute_tangent(self, evec, enorms):
        return tf.matrix_transpose((tf.matrix_transpose(evec) / enorms))

    def _compute_mass_matrix(self, radius, density, fullrestvl):
        mass = []
        dim = fullrestvl.get_shape().as_list()[0]
        for i in xrange(dim):
            m = density * math.pi * radius * radius * fullrestvl[i]
            mass.append(m)
            mass.append(m)
            mass.append(m)
        return tf.stack(mass)

    def _compute_bend_force(self):
        sqnorm = tf.reduce_sum(tf.multiply(self.kappa, self.kappa), 1, keep_dims=False)
        bend = tf.reduce_sum(tf.multiply(sqnorm, 1.0/self.innerrestvl))
        return -tf.gradients(bend, self.cpos)[0]

    def _compute_length_penalty_force(self, nrod):
        if self.clength is None:
            sq1 = nrod.enorms * nrod.enorms
            sq2 = self.restl * self.restl
            diff = sq1 - sq2
            potential = self.penalty_stiffness * diff * diff
            scalar = -tf.gradients(potential, nrod.enorms)[0]
            force = []
            # self.diff = diff
            # self.scalar = scalar

            dim = nrod.tan.get_shape().as_list()[0]
            force.append(-nrod.tan[0] * scalar[0])
            for i in xrange(1, dim):
                force.append(nrod.tan[i-1] * scalar[i-1] - nrod.tan[i] * scalar[i])
            force.append(nrod.tan[dim-1] * scalar[dim-1])

            self.clength = tf.concat(force, axis=0)
        return self.clength

    def update_cpos(self, h):
        return self.cpos + self.cvel * h

    def update_cvel(self, h):
        '''
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.cvel.get_shape(), dtype=tf.float32)
        '''
        # TODO: add lagrange multiplier here to cvel update
        bend_force = tf.reshape(self._compute_bend_force(), self.mass.get_shape())
        clength = self._compute_length_penalty_force(self.nrod)
        return self.cvel \
            + h * tf.reshape(self.mass * bend_force, self.cvel.get_shape()) \
            + h * tf.reshape(self.mass * clength, self.cvel.get_shape())

    def update_theta(self, h):
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.theta.get_shape(), dtype=tf.float32)

    def update_omega(self, h):
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.omega.get_shape(), dtype=tf.float32)

    def next_state(self, h):
        if self.nrod is None:
            self.nrod = ElasticRod(
                self.update_cpos(h), None,
                self.update_theta(h), None,
                self.restl)
            self.nrod.cvel = self.update_cvel(h)
            self.nrod.omega = self.update_omega(h)
        return self.nrod

    def dump(self, sess, feeds, name="rod"):
        print >> sys.stderr, "{rod} configuration".format(rod=name)
        print >> sys.stderr, "-" * 80
        print >> sys.stderr, "{rod}.cpos:\n{value}".format(rod=name,  value=sess.run(self.cpos, feed_dict=feeds))
        print >> sys.stderr, "{rod}.cvel:\n{value}".format(rod=name,  value=sess.run(self.cvel, feed_dict=feeds))
        print >> sys.stderr, "{rod}.theta:\n{value}".format(rod=name, value=sess.run(self.theta, feed_dict=feeds))
        print >> sys.stderr, "{rod}.omega:\n{value}".format(rod=name, value=sess.run(self.omega, feed_dict=feeds))
        print >> sys.stderr, "{rod}.evec:\n{value}".format(rod=name, value=sess.run(self.evec, feed_dict=feeds))
        print >> sys.stderr, "{rod}.enorms: {value}".format(rod=name, value=sess.run(self.enorms, feed_dict=feeds))
        print >> sys.stderr, "{rod}.kappa:\n{value}".format(rod=name, value=sess.run(self.kappa, feed_dict=feeds))
        print >> sys.stderr, "{rod}.restl: {value}".format(rod=name, value=sess.run(self.restl, feed_dict=feeds))
        print >> sys.stderr, "{rod}.fullrestvl: {value}".format(rod=name, value=sess.run(self.fullrestvl, feed_dict=feeds))
        print >> sys.stderr, "{rod}.innerrestvl: {value}".format(rod=name, value=sess.run(self.innerrestvl, feed_dict=feeds))
        print >> sys.stderr, "{rod}.mass:\n{value}".format(rod=name, value=sess.run(self.mass, feed_dict=feeds))
        # if self.clength is not None:
        #     print >> sys.stderr, "{rod}.tan:\n{value}".format(rod=name, value=sess.run(self.tan, feed_dict=feeds))
        #     reshape = tf.reshape(self.clength, shape=self.cvel.get_shape())
        #     print >> sys.stderr, "{rod}.clength:\n{value}".format(rod=name, value=sess.run(reshape, feed_dict=feeds))
        #     print >> sys.stderr, "{rod}.length diff: {value}".format(rod=name, value=sess.run(self.diff, feed_dict=feeds))
        #     print >> sys.stderr, "{rod}.diff scalar: {value}".format(rod=name, value=sess.run(self.scalar, feed_dict=feeds))
        print >> sys.stderr, "-" * 80
        return self
