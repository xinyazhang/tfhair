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

    def __init__(self, cpos, cvel, theta, omega, restl):
        # primary variables for phase update
        self.cpos = cpos
        self.cvel = cvel
        self.theta = theta
        self.omega = omega
        # constraints
        self.restl = restl
        self.restl_multiplier = tf.Variable(
            tf.zeros(self.restl.get_shape(), dtype=tf.float32))   # use Newton's method
        # auxiliary variables to help calculation
        self.evec = self._compute_edge_vector(self.cpos)
        self.enorms = self._compute_edge_norm(self.evec)
        self.tan = self._compute_tangent(self.evec, self.enorms)
        self.kappa = self._compute_curvature(self.evec, self.enorms)
        self.fullrestvl = self._compute_full_restvl(self.restl)
        self.innerrestvl = self._compute_inner_restvl(self.fullrestvl)

    def _compute_edge_vector(self, cpos):
        return cpos[1:,:] - cpos[0:-1,:]

    def _compute_edge_norm(self, evec):
        return tf.norm(evec, axis=1)

    def _compute_full_restvl(self, restl):
        en_i = tf.pad(restl, [[0, 0], [1, 0]], "CONSTANT")
        en_i_1 = tf.pad(restl, [[0, 0], [0, 1]], "CONSTANT")
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
        return evec / enorms

    def _compute_bend_force(self):
        sqnorm = tf.reduce_sum(tf.multiply(self.kappa, self.kappa), 1, keep_dims=False)
        bend = tf.reduce_sum(tf.multiply(sqnorm, 1.0/self.innerrestvl))
        return -tf.gradients(bend, self.cpos)[0]

    def _compute_length_constraint(self, nrod):
        # TODO: This should be called on nrod, since we want
        # primarily nrod to satisfy length constraints
        sq1 = tf.tensordot(nrod.restl, nrod.restl, axes=1)
        sq2 = tf.tensordot(nrod.enorms, nrod.enorms, axes=1)
        return sq1 - sq2

    def solve_length_constraint(self, nrod, sess, feeds):
        clength = self._compute_length_constraint(nrod)
        # TODO: use Newton's method to optimize lagrange multipliers

    def update_cpos(self, h):
        return self.cpos + self.cvel * h

    def update_cvel(self, h):
        '''
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.cvel.get_shape(), dtype=tf.float32)
        '''
        # TODO: add lagrange multiplier here to cvel update
        return self.cvel + h * self._compute_bend_force()
        # + self.restl_multiplier * ...

    def update_theta(self, h):
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.theta.get_shape(), dtype=tf.float32)

    def update_omega(self, h):
        # TODO: This is a placeholder before we actually
        # implemented velocity update
        return tf.zeros(self.omega.get_shape(), dtype=tf.float32)

    def dump(self, sess, feeds, name="rod"):
        print "{rod} configuration".format(rod=name)
        print "-" * 80
        print "{rod}.cpos:\n{value}".format(rod=name,  value=sess.run(self.cpos, feed_dict=feeds))
        print "{rod}.cvel:\n{value}".format(rod=name,  value=sess.run(self.cvel, feed_dict=feeds))
        print "{rod}.theta:\n{value}".format(rod=name, value=sess.run(self.theta, feed_dict=feeds))
        print "{rod}.omega:\n{value}".format(rod=name, value=sess.run(self.omega, feed_dict=feeds))
        print "{rod}.evec:\n{value}".format(rod=name, value=sess.run(self.evec, feed_dict=feeds))
        print "{rod}.enorms: {value}".format(rod=name, value=sess.run(self.enorms, feed_dict=feeds))
        print "{rod}.kappa:\n{value}".format(rod=name, value=sess.run(self.kappa, feed_dict=feeds))
        print "{rod}.fullrestvl: {value}".format(rod=name, value=sess.run(self.fullrestvl, feed_dict=feeds))
        print "{rod}.innerrestvl: {value}".format(rod=name, value=sess.run(self.innerrestvl, feed_dict=feeds))
        print "-" * 80
        return self
