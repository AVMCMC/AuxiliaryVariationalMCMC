import time
import os

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from LearningToSample.src.sampler import Sampler
from LearningToSample.src.L2HMC.dynamics import Dynamics
from LearningToSample.src.L2HMC.layers import Sequential, Zip, Parallel, ScaleTanh, Linear
from LearningToSample.src.eval import batch_means_ess, acceptance_rate

TF_FLOAT = tf.float32


class L2HMC(Sampler):

    def __init__(self, energy_function, arch, init_dist, tar_dim=2, scale=0.1,
                 leap_steps=10, leap_size=0.1, lambda_b=0.0, use_temp=True):
        self.energy_func = energy_function
        self.scale = scale
        self.init_dist = init_dist
        self.tar_dim = tar_dim
        self.lambda_b = lambda_b
        self.x = tf.placeholder(tf.float32, shape=[None, tar_dim])
        self.x_init = tf.placeholder(tf.float32, shape=[None, tar_dim])
        self.net_factory = self._get_net_factory(arch)
        self.dynamics = Dynamics(tar_dim, energy_function, leap_steps, leap_size,
                                 net_factory=self.net_factory,
                                 use_temperature=use_temp)
        self.sample_op, self.loss_op, self.acc = self._build_loss_and_sampler()

    def train(self, sess, learning_rate=1e-3, max_iters=5000,
              train_samples=200,
              save_path='.', log_freq=100, init_temp=1.0, temp_factor=1.0,
              temp_rate=100, save_freq=1000,
              train_plot=True, evaluate_steps=500, evaluate_burn_in=100,
              **kwargs):

        global_step = tf.Variable(0., name='global_step', trainable=False)
        if learning_rate < 0.0:
            u = np.random.uniform(2.5, 7)
            learning_rate = np.power(u, 10)
            if save_path:
                with open(
                        os.path.join(save_path['info'], 'params.txt'),
                        'a') as f:
                    f.write("learning_rate: {learning_rate} \n")
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   1000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(self.loss_op, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if save_path:
            ckpt = tf.train.get_checkpoint_state(save_path['ckpts'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        train_time = 0.0
        losses = []

        # Move attribute look ups outside loop to improve performance
        loss_op = self.loss_op
        sample_op = self.sample_op
        acc = self.acc
        x = self.x
        x_init = self.x_init
        init_dist = self.init_dist
        samples = init_dist(train_samples)
        init_samples = init_dist(train_samples)
        dynamics = self.dynamics

        for t in range(max_iters):
            tmp = (init_temp-1) * (1 - t / float(max_iters)) + 1
            time1 = time.time()
            _, loss_, samples, px_, lr_ = sess.run([
                train_op,
                loss_op,
                sample_op,
                acc,
                learning_rate
            ], {x: samples, dynamics.temperature: tmp,
                x_init: init_samples})
            time2 = time.time()
            init_samples = init_dist(train_samples)
            train_time += time2 - time1
            losses.append(loss_)

            if t % log_freq == 0:
                print(
                    'Time: %d Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f, LR: %.5f, Temp: %.4f' % (
                    train_time, t, max_iters, loss_, np.mean(px_), lr_, tmp))

                z, _ = self.sample(sess, evaluate_steps + evaluate_burn_in,
                                   temperature=1.0)

                z = z[evaluate_burn_in:]
                ess = batch_means_ess(z)
                min_ess = np.mean(np.min(ess, axis=1), axis=0)
                std_ess = np.std(np.min(ess, axis=1), axis=0)
                acc_rate = acceptance_rate(z)
                if save_path:
                    with open(
                            os.path.join(save_path['results'], 'results.txt'),
                            'a') as f:
                        f.write(
                            f"min ess at iteration {t}: {min_ess} +- {std_ess} \n")
                        f.write(f"acceptance at iteration {t}: {acc_rate} \n")
                else:
                    print(f"min ess at iteration {t}: {min_ess} +- {std_ess} \n ",
                          f"acceptance at iteration {t}: {acc_rate} \n")

                if save_path and train_plot and z.shape[2] == 2:
                    def plot2d(samples):
                        fig, ax = plt.subplots()
                        ax.scatter(samples[:, 0, 0], samples[:, 0, 1])
                        ax.set_aspect('equal', 'box')
                        plt.savefig(
                            os.path.join(save_path['figs'],
                                         f"samples_{t}.png"))
                        plt.close()

                    plot2d(z)

                if save_path and (t+1) % save_freq:
                    saver.save(sess, os.path.join(save_path['ckpts'], 'ckpt'),
                               global_step=global_step)

        return np.array(losses), train_time

    def sample(self, sess, num_samples, num_chains=10, temperature=1.0,
               **kwargs):

        init_samples = self.init_dist(num_chains)
        sample = init_samples
        samples = []

        # Remove the member attributes from the for loop because it hurts performance
        x = self.x
        x_init = self.x_init
        dynamics = self.dynamics
        sample_op = self.sample_op

        sample_time = 0.0
        for t in range(num_samples):
            start = time.time()
            feed_dict = {
                x: sample, dynamics.temperature: temperature,
            }
            sample = sess.run(sample_op, feed_dict)
            end = time.time()
            sample_time += end - start
            samples.append(sample)

        return np.array(samples), sample_time

    def _build_loss_and_sampler(self):
        with tf.name_scope('loss'):
            # The loss is a combination of a term averaged over samples from the
            # initial distribution and the target distribution

            def loss_helper(x, z):
                Lx, _, px, output = self._propose(x, self.dynamics, do_mh_step=True)
                Lz, _, pz, _ = self._propose(z, self.dynamics, do_mh_step=False)

                # Squared jumped distance
                v1 = (tf.reduce_sum(tf.square(x - Lx), axis=1) * px) + 1e-4
                v2 = (tf.reduce_sum(tf.square(z - Lz), axis=1) * pz) + 1e-4

                # Update loss
                loss = self.scale * (tf.reduce_mean(1.0 / v1) + tf.reduce_mean(1.0 / v2))
                loss += (- tf.reduce_mean(v1) - tf.reduce_mean(v2)) / self.scale
                return loss, output, px

            z = tf.random_normal(tf.shape(self.x))
            z_init = tf.random_normal(tf.shape(self.x))

            loss1, output, px = loss_helper(self.x, z)
            loss2, _, _ = loss_helper(self.x_init, z_init)

            loss = loss1 + self.lambda_b * loss2

        return output[0], loss, px

    def _propose(self, x, dynamics, init_v=None, aux=None, do_mh_step=False,
                log_jac=False):
        with tf.name_scope('propose'):
            if dynamics.hmc:
                Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
                return Lx, Lv, px, [self._tf_accept(x, Lx, px)]
            else:
                # sample mask for forward/backward
                mask = tf.cast(
                    tf.random_uniform((tf.shape(x)[0], 1), maxval=2, dtype=tf.int32),
                    TF_FLOAT)
                Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
                Lx2, Lv2, px2 = dynamics.backward(x, aux=aux, log_jac=log_jac)

            Lx = mask * Lx1 + (1 - mask) * Lx2

            Lv = None
            if init_v is not None:
                Lv = mask * Lv1 + (1 - mask) * Lv2

            px = tf.squeeze(mask, axis=1) * px1 + tf.squeeze(1 - mask, axis=1) * px2

            outputs = []

            if do_mh_step:
                outputs.append(self._tf_accept(x, Lx, px))

        return Lx, Lv, px, outputs

    def _tf_accept(self, x, Lx, px):
        mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
        return tf.where(mask, Lx, x)

    def _get_net_factory(self, arch):
        def network(x_dim, scope, factor):
            with tf.variable_scope(scope):
                net = Sequential([
                    Zip([
                        Linear(x_dim, arch[0], scope='embed_1', factor=1.0 / 3),
                        Linear(x_dim, arch[0], scope='embed_2',
                               factor=factor * 1.0 / 3),
                        Linear(2, arch[0], scope='embed_3', factor=1.0 / 3),
                        lambda _: 0.,
                    ]),
                    sum,
                    tf.nn.relu,
                    Linear(arch[0], arch[1], scope='linear_1'),
                    tf.nn.relu,
                    Parallel([
                        Sequential([
                            Linear(arch[1], x_dim, scope='linear_s', factor=0.001),
                            ScaleTanh(x_dim, scope='scale_s')
                        ]),
                        Linear(arch[1], x_dim, scope='linear_t', factor=0.001),
                        Sequential([
                            Linear(arch[1], x_dim, scope='linear_f', factor=0.001),
                            ScaleTanh(x_dim, scope='scale_f'),
                        ])
                    ])
                ])

            return net
        return network
