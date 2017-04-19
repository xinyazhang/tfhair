#!/usr/bin/env python2

import tensorflow as tf
import test_V
import test_T
import test_L
import test_C
import test_PT

#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())

# testi=tf.placeholder(tf.float32, shape=[1])
# testo=tf.gradients(-testi*testi, testi)[0]

# print(testo.eval(feed_dict={testi:[0]}))

test_PT.run()
test_V.run()
test_T.run()
test_L.run()
test_C.run()

