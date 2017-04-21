#!/usr/bin/env python2

import os
import shutil
import math
import numpy as np
import tensorflow as tf
import ElasticRod

import BlenderUtil

class RodSaver(object):

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.frame = 0
        self.connector = BlenderUtil.Sender()
        # clean dest directory
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.connector.send_finish()

    def add_timestep(self, cpos, theta):
        # save all rods into matrix
        n_rods = len(cpos)
        n_knots = cpos[0].shape[0]

        results = np.zeros(shape=(n_rods, n_knots, 4), dtype=np.float32)
        for j in xrange(n_rods):
            results[j,:,0:3] = cpos[j]
            results[j,:,  3] = np.reshape(theta[j], (4))
        filename = os.path.join(self.directory, "%d.npy" % self.frame)
        np.save(filename, results)
        self.connector.send_update(self.frame, filename)
        self.frame += 1
