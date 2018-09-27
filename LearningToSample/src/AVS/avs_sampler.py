import time
import os

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from LearningToSample.src.sampler import Sampler


class VariationalSampler(Sampler):

    def __init__(self, energy_func, target_dim, aux_dim, hdn_dim,
                 num_layers=3, train_samples=10, num_chains=10,
                 non_linearity='tanh', num_mix=1, perturb=1.0):
        self.target_logprob = energy_func
        self.non_linearity = self._get_act(non_linearity)
        self.num_layers = num_layers
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.hdn_dim = hdn_dim
        self.trn_smpls = train_samples
        self.num_chains = num_chains
        self.num_mix = num_mix
        self.perturb = perturb
        self._build_loss()
        self._build_sampler()

    def train(self, save_path=None, max_iters=10000, optimizer=tf.train.AdamOptimizer,
              sess=None, print_freq=100, save_freq=500,
              init=True, extra_phi_steps=0, learning_rate=1e-3,
              tol=0.2, **kwargs):
        """ Trains and returns total time to train"""

        if save_path:
            writer = tf.summary.FileWriter(save_path['logs'])

        if sess is None:
            sess = tf.Session()

        if init:
            if optimizer is not None:
                global_step1 = tf.Variable(0., name='global_step',
                                          trainable=False)
                global_step2 = tf.Variable(0., name='global_step',
                                           trainable=False)
                learning_rate1 = tf.train.exponential_decay(learning_rate,
                                                           global_step1, 1000,
                                                           0.96,
                                                           staircase=True)
                learning_rate2 = tf.train.exponential_decay(learning_rate,
                                                            global_step2, 1000,
                                                            0.96,
                                                            staircase=True)

                self.opt_op1 = optimizer(learning_rate=learning_rate1)
                self.opt_op2 = optimizer(learning_rate=learning_rate2)
                self.phi_op = self.opt_op1.minimize(self.loss,
                                                    var_list=self.phi_vars,
                                                    global_step=global_step1)
                self.theta_op = self.opt_op2.minimize(self.loss,
                                                      var_list=self.theta_vars,
                                                      global_step=global_step2)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=3)

        losses = [0.0]
        deltas = []
        train_time = 0.0

        # Get class attributes outside loop to improve performance
        loss = self.loss
        theta_op = self.theta_op
        phi_op = self.phi_op


        for t in range(max_iters):
            time1 = time.time()
            for i in range(extra_phi_steps):
                _ = sess.run(phi_op)
            loss_, _, _ = sess.run([loss, theta_op, phi_op])
            time2 = time.time()
            train_time += time2 - time1
            losses.append(loss_)
            deltas.append(abs(losses[-1] - losses[-2]))

            if t % print_freq == 0:
                print('{}:   loss:   {}'.format(t, loss_))
                sum_ = sess.run(self.loss_sum)
                writer.add_summary(sum_)

            if save_path and (t+1) % save_freq == 0:
                saver.save(sess, os.path.join(save_path['ckpts'], 'ckpt'))

            if t > 100:
                if np.mean(deltas[-100:]) < tol:
                    if save_path:
                        saver.save(sess,
                                   os.path.join(save_path['ckpts'], 'ckpt'),
                                   global_step=global_step1)
                    return losses, train_time

        if save_path:
            saver.save(sess, os.path.join(save_path['ckpts'], 'ckpt'),
                       global_step=global_step1)
        return losses, train_time



    def sample(self, num_samples, sess, writer, **kwargs):


        x_samps = []

        a = np.zeros((self.num_chains, self.aux_dim)).astype(np.float32)
        x_initialiser = self._aux_xgiva(tf.constant(a))
        x = sess.run(x_initialiser.sample())
        fetches = self.x_samples
        init_x_op = self.init_x

        sample_time = 0.0
        for i in range(num_samples):
            time1 = time.time()
            x = sess.run(fetches, feed_dict={init_x_op: x})
            time2 = time.time()
            sample_time += time2 - time1
            x_samps.append(x)

        return np.array(x_samps), sample_time


    def _build_loss(self):
        self.Qa = self._aux_a()
        self.A = self._aux_a().sample()
        self.Qxa = self._aux_xgiva(self.A)
        self.x_samples = self.Qxa.sample()
        self.Pax = self._aux_agivx(self.x_samples)

        self.loss = tf.reduce_mean(self.Qa.log_prob(self.A) +
                                   self.Qxa.log_prob(self.x_samples) -
                                   self.Pax.log_prob(self.A) -
                                   self.target_logprob(self.x_samples))

        self.loss_sum = tf.summary.scalar("loss", self.loss)
        # in case the class is created within a variable scope
        scope_name = tf.get_variable_scope().name

        self.phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=scope_name + '/aux_agivx')

        self.theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=scope_name + "/aux_xgiva")

    def _aux_a(self):
        with tf.variable_scope('aux_a', reuse=tf.AUTO_REUSE):
            aux_mean = tf.zeros([self.trn_smpls, self.aux_dim])
            Qa = tfd.MultivariateNormalDiag(loc=aux_mean)
        return Qa

    def _aux_xgiva(self, a):
        with tf.variable_scope('aux_xgiva', reuse=tf.AUTO_REUSE):
            h = a
            for _ in range(self.num_layers):
                h = tf.layers.dense(h, self.hdn_dim, activation=self.non_linearity)
            mean_and_sigma = tf.layers.dense(h, 2*self.target_dim)
            mean = mean_and_sigma[:, :self.target_dim]
            sigma = tf.exp(mean_and_sigma[:, self.target_dim:])
            Qxa = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        return Qxa

    def _aux_agivx(self, x_samples):
        """ Variational distribution is a mixture of diagonal gaussians with
        mixture components that are a function of the mean """
        with tf.variable_scope('aux_agivx', reuse=tf.AUTO_REUSE):
            h = x_samples
            for _ in range(self.num_layers):
                h = tf.layers.dense(h, self.hdn_dim, activation=self.non_linearity)
            mean_and_sigma = tf.layers.dense(h, 2*self.num_mix*self.aux_dim + self.num_mix)
            means = []
            sigmas = []
            for n in range(self.num_mix):
                begin = self.aux_dim * n
                end = begin + self.aux_dim
                offset = self.num_mix*self.aux_dim
                means.append(mean_and_sigma[:, begin: end])
                sigmas.append(tf.exp(mean_and_sigma[:, offset+begin: offset+end]))
            mix_comps = tf.exp(mean_and_sigma[:, - self.num_mix :])
            mix_comps = mix_comps/tf.reduce_sum(mix_comps, axis=1, keepdims=True)
            self.mix_comps = mix_comps  # just for plotting make this a class attribute
            Qax = tfd.Mixture(cat=tfd.Categorical(probs=mix_comps),
                              components=[tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
                              for m, s in zip(means, sigmas)])
        return Qax

    def _build_sampler(self):
        self.init_x = tf.placeholder(dtype=tf.float32,
                                     name="init_x",
                                     shape=[self.num_chains, self.target_dim])


        # Get the acceptance probability
        self._set_up_dists()
        self._get_aratio()

        u = tf.random_uniform(self.aratio.shape)
        accept = tf.less(tf.log(u), self.aratio)
        self.n_accept = tf.reduce_mean(tf.where(accept,
                                            tf.ones(accept.shape),
                                            tf.zeros(accept.shape)))
        tf.summary.scalar('num_accepted', self.n_accept)
        self.x_samples = tf.where(tf.squeeze(accept), self.prop_x, self.init_x)

    def _set_up_dists(self):
        self.smpPax_init = self._aux_agivx(self.init_x)
        self.a_samp = self.smpPax_init.sample()
        self.smpQxa_samp = self._aux_xgiva(self.a_samp)

        self.a_prime = tfd.MultivariateNormalDiag(loc=self.a_samp, scale_diag=tf.ones_like(self.a_samp)*self.perturb).sample()
        self.smpQxa_prime = self._aux_xgiva(self.a_prime)


        self.prop_x = self.smpQxa_prime.sample()
        self.smpPax_prop = self._aux_agivx(self.prop_x)


    def _get_aratio(self):
        self.aratio = (self.smpQxa_samp.log_prob(self.init_x) +
                       self.smpPax_prop.log_prob(self.a_prime) +
                       self.target_logprob(self.prop_x) -
                       self.smpQxa_prime.log_prob(self.prop_x) -
                       self.smpPax_init.log_prob(self.a_samp) -
                       self.target_logprob(self.init_x)
                      )
        tf.summary.scalar('aratio', tf.reduce_mean(self.aratio))

    def _get_act(self, non_lin):
        if non_lin == 'tanh':
            return tf.nn.tanh
        elif non_lin == 'relu':
            return tf.nn.relu
        else:
            raise ValueError('Non linearity must be tanh or relu')


class NaiveVariationalSampler(VariationalSampler):

    def _build_sampler(self):
        self.init_x = tf.placeholder(dtype=tf.float32,
                                     name="init_x",
                                     shape=[self.num_chains, self.target_dim])

        # Get the acceptance probability
        self._set_up_dists()
        self._get_aratio()

        u = tf.random_uniform(self.aratio.shape)
        accept = tf.less(tf.log(u), self.aratio)
        self.n_accept = tf.reduce_mean(tf.where(accept,
                                                tf.ones(accept.shape),
                                                tf.zeros(accept.shape)))
        tf.summary.scalar('num_accepted', self.n_accept)
        self.x_samples = tf.where(tf.squeeze(accept), self.prop_x, self.init_x)

    def _set_up_dists(self):
        self.Qa = tfd.MultivariateNormalDiag(tf.zeros([self.num_chains, self.aux_dim]))
        self.prop_a = self.Qa.sample()
        self.Qxa_prop = self._aux_xgiva(self.prop_a)
        self.prop_x = self.Qxa_prop.sample()

    def _get_aratio(self):
        self.aratio = (self.Qxa_prop.log_prob(self.init_x) +
                       self.target_logprob(self.prop_x) +
                       self.target_logprob(self.init_x) -
                       self.Qxa_prop.log_prob(self.prop_x))

        tf.summary.scalar('aratio', tf.reduce_mean(self.aratio))


class NaiveJointVariationalSampler(VariationalSampler):

    def sample_step(self, sess, init_x, init_a, summ):
        fetches = [self.x_samples, self.a_samples, self.n_accept, self.aratio, summ]
        feed_dict = {self.init_x: init_x, self.init_a: init_a}

        # results is a tuple of the outcomes of fetches
        results = sess.run(fetches, feed_dict=feed_dict)

        return results

    def sample(self, num_samples, sess, writer, **kwargs):
        summ = tf.summary.merge_all()
        x_samps = []

        a = np.zeros((self.num_chains, self.aux_dim)).astype(np.float32)
        x_initialiser = self._aux_xgiva(tf.constant(a))
        x = sess.run(x_initialiser.sample())

        sample_time = 0.0
        for i in range(num_samples):
            time1 = time.time()
            x, a,  n_accept, ratio, summ_ = self.sample_step(sess, x, a, summ)
            time2 = time.time()
            sample_time += time2 - time1
            x_samps.append(x)
            writer.add_summary(summ_)

        return np.array(x_samps), sample_time

    def _build_sampler(self):
        self.init_x = tf.placeholder(dtype=tf.float32,
                                     name="init_x",
                                     shape=[self.num_chains, self.target_dim])

        self.init_a = tf.placeholder(dtype=tf.float32,
                                     name="init_a",
                                     shape=[self.num_chains, self.aux_dim])

        # Get the acceptance probability
        self._set_up_dists()
        self._get_aratio()

        u = tf.random_uniform(self.aratio.shape)
        accept = tf.less(tf.log(u), self.aratio)
        self.n_accept = tf.reduce_mean(tf.where(accept,
                                                tf.ones(accept.shape),
                                                tf.zeros(accept.shape)))
        tf.summary.scalar('num_accepted', self.n_accept)
        self.x_samples = tf.where(tf.squeeze(accept), self.prop_x, self.init_x)
        self.a_samples = tf.where(tf.squeeze(accept), self.prop_a, self.init_a)

    def _set_up_dists(self):
        self.Qa = tfd.MultivariateNormalDiag(tf.zeros([self.num_chains, self.aux_dim]))
        self.prop_a = self.Qa.sample()
        self.Qxa_init = self._aux_xgiva(self.init_a)
        self.Pax_init = self._aux_agivx(self.init_x)
        self.Qxa_prop = self._aux_xgiva(self.prop_a)
        self.prop_x = self.Qxa_prop.sample()
        self.Pax_prop = self._aux_agivx(self.prop_x)

    def _get_aratio(self):
        self.aratio = (self.Qxa_init.log_prob(self.init_x) +
                       self.Qa.log_prob(self.init_a) +
                       self.target_logprob(self.prop_x) +
                       self.Pax_prop.log_prob(self.prop_a) -
                       self.Pax_init.log_prob(self.init_a) -
                       self.Qa.log_prob(self.prop_a) -
                       self.target_logprob(self.init_x) -
                       self.Qxa_prop.log_prob(self.prop_x))

        tf.summary.scalar('aratio', tf.reduce_mean(self.aratio))


class JointVariationalSampler(VariationalSampler):


    def sample_step(self, sess, init_x, init_a, summ):
        fetches = [self.x_samples, self.a_samples, self.n_accept, self.aratio, summ]
        feed_dict = {self.init_x: init_x, self.init_a: init_a}

        # results is a tuple of the outcomes of fetches
        results = sess.run(fetches, feed_dict=feed_dict)

        return results

    def sample(self, num_samples, sess, writer, **kwargs):
        summ = tf.summary.merge_all()
        x_samps = []

        a = np.zeros((self.num_chains, self.aux_dim)).astype(np.float32)
        x_initialiser = self._aux_xgiva(tf.constant(a))
        x = sess.run(x_initialiser.sample())

        sample_time = 0.0
        for i in range(num_samples):
            time1 = time.time()
            x, a,  n_accept, ratio, summ_ = self.sample_step(sess, x, a, summ)
            time2 = time.time()
            sample_time += time2 - time1
            x_samps.append(x)
            writer.add_summary(summ_)

        return np.array(x_samps), sample_time

    def _build_sampler(self):
        self.init_x = tf.placeholder(dtype=tf.float32,
                                     name="init_x",
                                     shape=[self.num_chains, self.target_dim])

        self.init_a = tf.placeholder(dtype=tf.float32,
                                     name="init_a",
                                     shape=[self.num_chains, self.aux_dim])

        # Get the acceptance probability
        self._set_up_dists()
        self._get_aratio()

        u = tf.random_uniform(self.aratio.shape)
        accept = tf.less(tf.log(u), self.aratio)
        self.n_accept = tf.reduce_mean(tf.where(accept,
                                                tf.ones(accept.shape),
                                                tf.zeros(accept.shape)))
        tf.summary.scalar('num_accepted', self.n_accept)
        self.x_samples = tf.where(tf.squeeze(accept), self.prop_x, self.init_x)
        self.a_samples = tf.where(tf.squeeze(accept), self.prop_a, self.init_a)

    def _set_up_dists(self):
        self.Qxa_init = self._aux_xgiva(self.init_a)
        self.Pax_init = self._aux_agivx(self.init_x)
        self.prop_a = self.Pax_init.sample()
        self.Qxa_prop = self._aux_xgiva(self.prop_a)
        self.prop_x = self.Qxa_prop.sample()
        self.Pax_prop = self._aux_agivx(self.prop_x)

    def _get_aratio(self):
        self.aratio = (self.Qxa_init.log_prob(self.init_x) +
                       self.Pax_prop.log_prob(self.init_a) +
                       self.Pax_prop.log_prob(self.prop_a) +
                       self.target_logprob(self.prop_x) -
                       self.Qxa_prop.log_prob(self.prop_x) -
                       self.Pax_init.log_prob(self.prop_a) -
                       self.Pax_init.log_prob(self.init_a) -
                       self.target_logprob(self.init_x))
        tf.summary.scalar('aratio', tf.reduce_mean(self.aratio))