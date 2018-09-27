""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.AVS.avs_sampler import NaiveVariationalSampler as VS
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import MixtureOfGaussians


def main(args):

    print('setting up distribution')
    dist = MixtureOfGaussians(means=args.means,
                              stds=args.stds,
                              pis=[0.5, 0.5])

    print('setting up sampler')
    with tf.variable_scope('sampler', reuse=tf.AUTO_REUSE):
        sampler = VS(dist.log_prob_func(),
                     args.data_dim,
                     args.aux_dim,
                     args.hidden_units,
                     args.num_layers,
                     args.train_samples,
                     args.num_chains,
                     args.activation,
                     args.num_mix,
                     args.perturb)

        print('setting up and running experiment')
        exp = Experiment(log_dir=args.logdir,
                         sampler=sampler,
                         params=vars(args),
                         dist=dist)

        exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='logs/avs_ind/mog')

    parser.add_argument('--means', nargs='+', type=list,
                        default=[[10.0, 0.0], [-10.0, 0.0]], help='mog means')

    parser.add_argument('--stds', nargs='+', default=[[1.0, 1.0],[1.0, 1.0]],
                        type=list, help='mog stds')

    parser.add_argument('--perturb', default=1.0, type=float)

    parser.add_argument('--train_samples', type=int, default=200,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")

    parser.add_argument('--num_mix', type=int, default=1,
                        help="""Mixture components within variational approximation""")

    parser.add_argument('--aux_dim', type=int, default=1,
                        help="Dimensionality of lower dimensional space")

    parser.add_argument("--data_dim", type=int, default=2)

    parser.add_argument("--max_iters", type=int, default=600)

    parser.add_argument("--hidden_units", type=int, default=300,
                        help="""Number of units per layer in sampler""")

    parser.add_argument("--activation", type=str, default='tanh',
                        help="sampler activation function")

    parser.add_argument('--num_layers', type=int, default=2,
                        help="number of layers in sampler")

    parser.add_argument('--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument('--tol', type=float, default=0.2,
                        help="mean change in loss over 100 iterations below which we termiante training")
    args = parser.parse_args()
    main(args)