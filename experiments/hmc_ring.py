""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.HMC.hmc import HamiltonianMonteCarloSampler as HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import Ring


def main(args):

    def init_dist(bs):
        return np.random.randn(bs, args.data_dim).astype(np.float32)

    print('setting up distribution')
    dist = Ring(args.scale)

    print('setting up sampler')
    sampler = HMC(dist.log_prob_func(),
                  init_dist,
                  args.step_size,
                  args.num_leap_steps,
                  args.data_dim
                  )


    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',type=float, default=0.02)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--num_leap_steps', type=int, default=20)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/hmc/ring')
    parser.add_argument('-dd',"--data_dim", type=int, default=2)
    parser.add_argument('-ns', '--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")
    args = parser.parse_args()
    main(args)
