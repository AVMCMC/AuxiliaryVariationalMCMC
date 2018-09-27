""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.L2HMC.l2hmc_sampler import L2HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import FeedForwardNetPost


def main(args):

    print('setting up distribution')

    def load_data(folder, test_frac=0.2):
        X = np.load(folder + '/data.npy')
        y = np.load(folder + '/labels.npy')
        N, D = X.shape
        data = np.concatenate([X, y], axis=1)
        np.random.shuffle(data)
        train_data = data[:int(N * (1 - test_frac))]
        test_data = data[int(N * (1 - test_frac)):]
        return train_data[:, :-1], train_data[:, -1], test_data[:,
                                                      :-1], test_data[:, -1]

    X_train, y_train, X_test, y_test = load_data(args.data)
    data_dim = X_train.shape[1] + 1
    arch = [data_dim] + args.dist_arch
    target_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
    dist = FeedForwardNetPost(X_train, y_train, X_test, y_test, arch,
                              prec=args.prior)



    print('setting up sampler')
    def init_dist(bs):
        return np.random.randn(bs, target_dim)

    sampler = L2HMC(log_prob_func=dist.log_prob_func(),
                    arch=args.arch,
                    sample_dim=target_dim,
                    scale=args.scale,
                    init_dist=init_dist,
                    leap_steps=args.leap_steps,
                    eps=args.eps
                    )



    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/boston', type=str,
                        help=" Data directory where data is stored")
    parser.add_argument('--prior', default=0.1, type=float,
                        help="the variance of the normal prior placed on"
                             "the logistic regression parameters")
    parser.add_argument('--dist_arch', default=[50, 1], type=list,
                        help="layer widths for the bayesian net")
    parser.add_argument('--init_temp', default=1.0, type=float)
    parser.add_argument('--temp_factor', type=float, default=1.0)
    parser.add_argument('--temp_rate', default=100, type=int)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='logs/l2hmc/net')
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--leap_steps', default=10, type=int)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--arch', default=[100, 100], type=list)

    parser.add_argument('--train_samples', type=int, default=200,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")
    parser.add_argument("--max_iters", type=int, default=20000)

    parser.add_argument('--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")

    parser.add_argument('--num_chains', type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")

    parser.add_argument("--train_plot", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()
    main(args)