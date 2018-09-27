""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import numpy as np

from LearningToSample.src.RWM.rwm import RwmSampler
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import Ring


def main(args):

    print('setting up distribution')
    dist = Ring(args.scale)

    print('setting up sampler')

    def init_dist(bs):
        o = np.ones((bs, 1))
        a = np.array([[0.75, 0.75]])
        return o*a

    sampler = RwmSampler(dist.log_prob_func(),
                         args.data_dim,
                         init_dist,
                         args.perturb)

    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',type=float, default=0.005)
    parser.add_argument('--perturb', type=float, default=10.0)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/rwm/ring')
    parser.add_argument('-dd',"--data_dim", type=int, default=2)
    parser.add_argument('-ns', '--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    args = parser.parse_args()
    main(args)
