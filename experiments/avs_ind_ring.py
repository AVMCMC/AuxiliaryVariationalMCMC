""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf

from LearningToSample.src.AVS.avs_sampler import NaiveVariationalSampler as VS
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import Ring


def main(args):

    print('setting up distribution')
    dist = Ring(args.scale)

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
    parser.add_argument('--scale',type=float, default=0.02)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/avs_ind/ring')
    parser.add_argument('-t', '--train_samples', type=int, default=200,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")

    parser.add_argument('--perturb', type=float, default=1.0)

    parser.add_argument('-nm', '--num_mix', type=int, default=2,
                        help="""Mixture components within variational approximation""")

    parser.add_argument('-ad', '--aux_dim', type=int, default=1,
                        help="Dimensionality of lower dimensional space")

    parser.add_argument('-dd',"--data_dim", type=int, default=2)

    parser.add_argument('-mi', "--max_iters", type=int, default=3000)

    parser.add_argument('-hdn', "--hidden_units", type=int, default=300,
                        help="""Number of units per layer in sampler""")

    parser.add_argument('-act', "--activation", type=str, default='tanh',
                        help="sampler activation function")

    parser.add_argument('-nl', '--num_layers', type=int, default=2,
                        help="number of layers in sampler")

    parser.add_argument('-ns', '--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)


    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")

    parser.add_argument('--tol', type=float, default=0.2,
                        help="mean change in loss over 100 iterations below which we termiante training")
    args = parser.parse_args()
    main(args)
