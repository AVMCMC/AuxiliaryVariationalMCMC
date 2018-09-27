""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import tensorflow as tf
import numpy as np

from LearningToSample.src.HMC.hmc import HamiltonianMonteCarloSampler as HMC
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
        train_data = data[:int(N*(1-test_frac))]
        test_data = data[int(N*(1-test_frac)):]
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

    X_train, y_train, X_test, y_test = load_data(args.data)
    data_dim = X_train.shape[1] + 1
    arch = [data_dim] + args.arch
    target_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
    dist = FeedForwardNetPost(X_train, y_train, X_test, y_test, arch, prec=args.prior)

    def init_dist(bs):
        return np.random.normal(loc=0.0, scale=1.0, size=(bs, target_dim)).astype(np.float32)


    print('setting up sampler')
    sampler = HMC(dist.log_prob_func(),
                  init_dist,
                  args.step_size,
                  args.num_leap_steps,
                  target_dim
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
    parser.add_argument('--arch', default=[50, 1], type=list,
                        help="layer widths for the bayesian net")
    parser.add_argument('--step_size', type=float, default=0.0001)
    parser.add_argument('--num_leap_steps', type=int, default=100)
    parser.add_argument('--burn_in', type=int, default=10000)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/hmc/net')
    parser.add_argument('-dd',"--data_dim", type=int, default=2)
    parser.add_argument('-ns', '--num_samples', type=int, default=100000,
                        help="Number of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                        help="initial learning rate")
    args = parser.parse_args()
    main(args)
