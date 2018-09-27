""" Metropolis Hastings Sampler with symmetric gaussian proposal """
import time

import numpy as np
import tensorflow as tf

from LearningToSample.src.sampler import Sampler


class RwmSampler(Sampler):

    def __init__(self, log_prob, data_dim, init_dist, perturb=0.1):
        self.log_prob = log_prob
        self.init_dist = init_dist
        self.init_x = tf.placeholder(dtype=tf.float32, shape=[None, data_dim])
        self.scale = tf.constant(perturb)
        self.num_samples = tf.placeholder(dtype=tf.int32)

        def fn(x, step):
            prop_x = self._get_prop(x, self.scale)
            accept = tf.squeeze(self._accept(x, prop_x))
            x_ = tf.where(accept, prop_x, x)
            return x_

        elems = tf.zeros([self.num_samples])
        self.samples = tf.scan(fn, elems,
                               initializer=self.init_x)

    def sample(self, sess, num_samples=10000, num_chains=10, **kwargs):
        init_x = self.init_dist(num_chains)
        time1 = time.time()
        samples = sess.run(self.samples,
                           feed_dict={self.init_x: init_x,
                           self.num_samples: num_samples})
        time2 = time.time()
        return samples, time2 - time1

    def _get_prop(self, x, scale):
        return x + tf.random_normal(shape=tf.shape(x), stddev=scale)

    def _accept(self, x, prop_x):
        loga = tf.expand_dims(self.log_prob(prop_x), 1) - tf.expand_dims(self.log_prob(x), 1)
        accept = tf.less_equal(tf.log(tf.random_uniform(shape=tf.shape(loga), maxval=1)), loga)
        return accept

    def train(self, *args, **kwargs):
        return [0], 0


