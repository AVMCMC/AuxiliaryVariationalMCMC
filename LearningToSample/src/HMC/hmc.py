import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



class HamiltonianMonteCarloSampler(object):
    """
    Thin wrapper around tfp HMC implementation
    """
    def __init__(self, log_prob_fun, init_dist, stepsize=0.1, n_steps=10, data_dim=2):

        self.init_dist = init_dist
        self.current_state = tf.placeholder(dtype=tf.float32, shape=[None, data_dim])
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(log_prob_fun,
                                                         stepsize,
                                                         n_steps,
                                                         state_gradients_are_stopped=True)


    def sample(self, sess, num_samples, num_chains, **kwargs):
        samples, _ = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=0,
            current_state=self.current_state,
            kernel=self.hmc_kernel
        )
        time1 = time.time()
        samples_ = sess.run(samples, feed_dict={self.current_state:
                                                self.init_dist(num_chains)})
        time2 = time.time()
        return samples_, time2 - time1


    def train(self, *args, **kwargs):
        return [0], [0]



def kinetic_energy(v):
    return 0.5 * tf.reduce_sum(tf.multiply(v, v), axis=1)


def hamiltonian(p, v, f):
    """
    Return the value of the Hamiltonian
    :param p: position variable
    :param v: velocity variable
    :param f: energy function
    :return: hamiltonian
    """
    return f(p) + kinetic_energy(v)
#
#
def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Run Metropolis-Hastings algorithm for 1 step
    :param energy_prev:
    :param energy_next:
    :return: Tensor of boolean values, indicating accept or reject
    """
    energy_diff = energy_prev - energy_next
    return (tf.exp(energy_diff) - tf.random_uniform(tf.shape(energy_prev))) >= 0.0
#
#
# def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
#     def leapfrog(pos, vel, step, i):
#         de_dp_ = tf.gradients(tf.reduce_sum(energy_fn(pos)), pos)[0]
#         new_vel_ = vel - step * de_dp_
#         new_pos_ = pos + step * new_vel_
#         return [new_pos_, new_vel_, step, tf.add(i, 1)]
#
#     def condition(pos, vel, step, i):
#         return tf.less(i, n_steps)
#
#     de_dp = tf.gradients(tf.reduce_sum(energy_fn(initial_pos)), initial_pos)[0]
#     vel_half_step = initial_vel - 0.5 * stepsize * de_dp
#     pos_full_step = initial_pos + stepsize * vel_half_step
#
#     i = tf.constant(0)
#     final_pos, new_vel, _, _ = tf.while_loop(condition, leapfrog, [pos_full_step, vel_half_step, stepsize, i])
#     de_dp = tf.gradients(tf.reduce_sum(energy_fn(final_pos)), final_pos)[0]
#     final_vel = new_vel - 0.5 * stepsize * de_dp
#     return final_pos, final_vel
#
#
# def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
#     initial_vel = tf.random_normal(tf.shape(initial_pos))
#     final_pos, final_vel = simulate_dynamics(
#         initial_pos=initial_pos,
#         initial_vel=initial_vel,
#         stepsize=stepsize,
#         n_steps=n_steps,
#         energy_fn=energy_fn
#     )
#     accept = metropolis_hastings_accept(
#         energy_prev=hamiltonian(initial_pos, initial_vel, energy_fn),
#         energy_next=hamiltonian(final_pos, final_vel, energy_fn)
#     )
#     return accept, final_pos, final_vel
#
#
# def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept,
#                 target_acceptance_rate, stepsize_inc, stepsize_dec,
#                 stepsize_min, stepsize_max, avg_acceptance_slowness):
#     new_pos = tf.where(accept, final_pos, initial_pos)
#     new_stepsize_ = tf.multiply(
#         stepsize,
#         tf.where(tf.greater(avg_acceptance_rate, target_acceptance_rate), stepsize_inc, stepsize_dec)
#     )
#     new_stepsize = tf.maximum(tf.minimum(new_stepsize_, stepsize_max), stepsize_min)
#     new_acceptance_rate = tf.add(
#         avg_acceptance_slowness * avg_acceptance_rate,
#         (1.0 - avg_acceptance_slowness) * tf.reduce_mean(tf.to_float(accept))
#     )
#     return new_pos, new_stepsize, new_acceptance_rate
#
#
# class HamiltonianMonteCarloSampler(object):
#     """
#     TensorFlow implementation for Hamiltonian Monte Carlo
#     """
#     def __init__(self, energy_fn, prior, stepsize=0.1, n_steps=10,
#                  target_acceptance_rate=0.65, avg_acceptance_slowness=0.9,
#                  stepsize_min=0.001, stepsize_max=1000.0, stepsize_dec=0.97, stepsize_inc=1.03,
#                  inter_op_parallelism_threads=1, intra_op_parallelism_threads=1,
#                  sess=False):
#         self.energy_fn = energy_fn
#         self.prior = prior
#         self.z = tf.placeholder(dtype=tf.float32, name='z_hmc')
#         self.stepsize = tf.constant(stepsize)
#         self.avg_acceptance_rate = tf.constant(target_acceptance_rate)
#         self.sess = sess
#
#         def fn(zsa, x):
#             z, s, a = zsa
#             accept, final_pos, final_vel = hmc_move(
#                 z,
#                 energy_fn,
#                 s,
#                 n_steps
#             )
#             z_, s_, a_ = hmc_updates(
#                 z,
#                 s,
#                 avg_acceptance_rate=a,
#                 final_pos=final_pos,
#                 accept=accept,
#                 stepsize_min=stepsize_min,
#                 stepsize_max=stepsize_max,
#                 stepsize_dec=stepsize_dec,
#                 stepsize_inc=stepsize_inc,
#                 target_acceptance_rate=target_acceptance_rate,
#                 avg_acceptance_slowness=avg_acceptance_slowness
#             )
#             return z_, s_, a_
#
#         self.steps = tf.placeholder(tf.int32, [])
#         elems = tf.zeros([self.steps])
#
#         self.z_, self.stepsize_, self.avg_acceptance_rate_ = tf.scan(
#             fn, elems,
#             (self.z, self.stepsize, self.avg_acceptance_rate),
#             back_prop=False
#         )
#
#         if not self.sess:
#             # only want to start a session if running this independently
#             self.sess = tf.Session(
#                 config=tf.ConfigProto(
#                     inter_op_parallelism_threads=inter_op_parallelism_threads,
#                     intra_op_parallelism_threads=intra_op_parallelism_threads
#                 )
#             )
#             self.sess.run(tf.global_variables_initializer())
#
#     def sample(self, steps, batch_size):
#         z, stepsize, acceptance_rate = self.sess.run(
#             [self.z_, self.stepsize_, self.avg_acceptance_rate_],
#             feed_dict={self.steps: steps, self.z: self.prior(batch_size)}
#         )
#         return z
