""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.HMC.hmc import HamiltonianMonteCarloSampler as HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import MixtureOfGaussians


def main(args):


    print('setting up distribution')
    dist = MixtureOfGaussians(means=args.means,
                              stds=args.stds,
                              pis=[0.5, 0.5])


    def init_dist(bs):
        return np.random.randn(bs, 2).astype(np.float32)

    print('setting up sampler')
    sampler = HMC(dist.log_prob_func(),
                  init_dist,
                  args.step_size,
                  args.num_leap_steps,
                  2
                  )


    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--means', nargs='+', type=list,
                        default=[[10.0, 0.0], [-10.0, 0.0]], help='mog means')

    parser.add_argument('--stds', nargs='+', default=[[1.0, 1.0],[1.0, 1.0]],
                        type=list, help='mog stds')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--num_leap_steps', type=int, default=20)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/hmc/mog')
    parser.add_argument('-dd',"--data_dim", type=int, default=2)
    parser.add_argument('-ns', '--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    args = parser.parse_args()
    main(args)
