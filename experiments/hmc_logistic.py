""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.HMC.hmc import HamiltonianMonteCarloSampler as HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import BayesLogRegPost


def main(args):

    print(args)
    print('setting up distribution')
    def load_data(folder, test_frac=0.2):
        X = np.load(folder + '/data.npy')
        y = np.load(folder + '/labels.npy')
        N, D = X.shape
        data = np.concatenate([X, y], axis=1)
        np.random.shuffle(data)
        train_data = data[:int(N * (1 - test_frac))]
        test_data = data[int(N * (1 - test_frac)):]
        return train_data[:, :-1], train_data[:, -1], test_data[:,:-1], test_data[:, -1]

    X_train, y_train, X_test, y_test = load_data(args.data)
    data_dim = X_train.shape[1] + 1
    dist = BayesLogRegPost(X_train, y_train, X_test, y_test, args.prior)

    def init_dist(bs):
        return np.random.randn(bs, data_dim).astype(np.float32)

    print('setting up sampler')
    sampler = HMC(dist.log_prob_func(),
                  init_dist,
                  args.step_size,
                  args.num_leap_steps,
                  data_dim
                  )


    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/heart', type=str,
                        help=" Data directory where data is stored")
    parser.add_argument('--prior', default=0.1, type=float,
                        help="the variance of the normal prior placed on"
                             "the logistic regression parameters")
    parser.add_argument('--step_size', type=float, default=0.05)
    parser.add_argument('--num_leap_steps', type=int, default=20)
    parser.add_argument('--burn_in', type=int, default=0)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/hmc/logistic/heart')
    parser.add_argument('-dd',"--data_dim", type=int, default=2)
    parser.add_argument('-ns', '--num_samples', type=int, default=5000,
                        help="Number of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")
    args = parser.parse_args()
    main(args)
