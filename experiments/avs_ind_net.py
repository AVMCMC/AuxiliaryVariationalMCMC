""" Experiment to sample from the posterior of a Logistic Regression"""
import argparse

import numpy as np
import tensorflow as tf


from LearningToSample.src.distributions import FeedForwardNetPost
from LearningToSample.src.AVS.avs_sampler import VariationalSampler as VS
from LearningToSample.src.experiment import Experiment


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

    print('setting up sampler')
    with tf.variable_scope('sampler', reuse=tf.AUTO_REUSE):
        sampler = VS(dist.log_prob_func(),
                     target_dim,
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
    parser.add_argument('--data', default='data/boston', type=str,
                        help=" Data directory where data is stored")
    parser.add_argument('--prior', default=0.1, type=float,
                        help="the variance of the normal prior placed on"
                             "the logistic regression parameters")
    parser.add_argument('--arch', default=[50, 1], type=list,
                        help="layer widths for the bayesian net")
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--perturb', type=float, default=1.0)
    parser.add_argument('--logdir', type=str,
                        default='logs/avs_ind/net')
    parser.add_argument('--train_samples', type=int, default=10,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")
    parser.add_argument('--num_mix', type=int, default=1,
                        help="""Mixture components within variational approximation""")
    parser.add_argument('--aux_dim', type=int, default=30,
                        help="Dimensionality of lower dimensional space")
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--hidden_units", type=int, default=500,
                        help="""Number of units per layer in sampler""")
    parser.add_argument("--activation", type=str, default='relu',
                        help="sampler activation function")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="number of layers in sampler")
    parser.add_argument('--num_samples', type=int, default=25000,
                        help="Numer of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-5,
                        help="initial learning rate")
    parser.add_argument('--tol', type=float, default=0.2,
                        help="mean change in loss over 100 iterations below which we termiante training")
    args = parser.parse_args()
    main(args)
