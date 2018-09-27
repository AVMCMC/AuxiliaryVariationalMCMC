""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.L2HMC.l2hmc_sampler import L2HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import BayesLogRegPost


def main(args):
    # If negative parameters provided, select them at random from a suitable range
    if args.learning_rate < 0:
        u = np.random.uniform(2.5, 6)
        args.learning_rate = np.float32(np.power(10.0, -u))
    if args.scale < 0:
        args.scale = np.float32(np.random.uniform(0.1, 5.0))
    if args.lambda_b < 0:
        args.lambda_b = np.float32(np.random.uniform(0.0, 0.5))
    if args.eps < 0:
        u = np.random.uniform(1, 2)
        args.eps = np.float32(np.power(10.0, -u))
    if args.leap_steps < 0.0:
        args.leap_steps = np.random.choice([10, 25, 50])

    print(args)

    print('setting up distribution')
    def load_data(folder, test_frac=0.2):
        X = np.load(folder + '/data.npy')
        y = np.load(folder + '/labels.npy')
        N, D = X.shape
        data = np.concatenate([X, y], axis=1)
        np.random.shuffle(data)
        train_data = data[:int(N*(1-test_frac))]
        test_data = data[int(N*(1-test_frac)):]
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

    X_train, y_train, X_test, y_test = load_data(args.data)
    data_dim = X_train.shape[1] + 1
    with tf.name_scope('distribution'):
        dist = BayesLogRegPost(X_train, y_train, X_test, y_test, args.prior)
    energy = lambda x: - dist.log_prob_func()(x)



    print('setting up sampler')
    def init_dist(bs):
        return np.random.randn(bs, data_dim)

    with tf.name_scope('sampler'):
        sampler = L2HMC(energy_function=energy,
                        arch=args.arch,
                        tar_dim=data_dim,
                        scale=args.scale,
                        init_dist=init_dist,
                        leap_steps=args.leap_steps,
                        leap_size=args.eps,
                        lambda_b=args.lambda_b
                        )



    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist,
                     debug=args.debug)

    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--data', default='data/german', type=str,
                        help=" Data directory where data is stored")
    parser.add_argument('--prior', default=0.1, type=float,
                        help="the variance of the normal prior placed on"
                             "the logistic regression parameters")
    parser.add_argument('--init_temp', default=1.0, type=float)
    parser.add_argument('--lambda_b', default=-0.5, type=float,
                        help='factor determining trade-off between fast-mixing and burn-in')
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='logs/l2hmc/logistic/german')
    parser.add_argument('--scale', default=-1.0, type=float)
    parser.add_argument('--leap_steps', default=10, type=int)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--arch', default=[100, 100], type=list)

    parser.add_argument('--train_samples', type=int, default=200,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")
    parser.add_argument("--max_iters", type=int, default=10000)

    parser.add_argument('--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")

    parser.add_argument('--num_chains', type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")

    parser.add_argument("--train_plot", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()
    main(args)