""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import numpy as np


from LearningToSample.src.L2HMC.l2hmc_sampler import L2HMC
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.L2HMC.distributions import GMM


def main(args):
    if args.learning_rate < 0:
        u = np.random.uniform(2.5, 6)
        args.learning_rate = np.float32(np.power(10, -u))
    if args.scale < 0:
        args.scale = np.float32(np.random.uniform(0.1, 5.0))
    if args.lambda_b < 0:
        args.lambda_b = np.float32(np.random.uniform(0.0, 0.5))
    if args.leap_size< 0:
        u = np.random.uniform(1, 2)
        args.leap_size = np.float32(np.power(10, -u))
    if args.leap_steps < 0.0:
        args.leap_steps = np.random.choice([10, 25, 50])
    if args.init_temp < 0.0:
        args.init_temp = np.random.choice([100, 50])

    print(args)

    print('setting up distribution')
    mu_1 = np.array([-10., 0.])
    mu_2 = np.array([10., 0.])
    mus = np.array([mu_1, mu_2])
    cov_1 = 1 * np.eye(2)
    cov_2 = 1 * np.eye(2)
    covs = np.array([cov_1, cov_2])
    pis = np.array([0.5, 0.5])
    dist = GMM(mus, covs, pis)



    def init_dist(bs):
        return np.random.randn(bs, args.data_dim)

    sampler = L2HMC(energy_function=dist.get_energy_function(),
                    arch=args.arch,
                    tar_dim=args.data_dim,
                    scale=args.scale,
                    init_dist=init_dist,
                    leap_steps=args.leap_steps,
                    leap_size=args.leap_size,
                    lambda_b=args.lambda_b,
                    use_temp=True
                    )



    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist)

    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--init_temp', default=-100.0, type=float)
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='logs/l2hmc/mog')

    parser.add_argument('--means', nargs='+', type=list,
                        default=[[10.0, 0.0], [-10.0, 0.0]], help='mog means')

    parser.add_argument('--stds', nargs='+', default=[[1.0, 1.0],[1.0, 1.0]],
                        type=list, help='mog stds')

    parser.add_argument('--scale', default=-0.1, type=float)
    parser.add_argument('--lambda_b', default=-0.2, type=float,
                        help='factor determining trade-off between fast-mixing and burn-in')
    parser.add_argument('--leap_steps', default=-10, type=int)
    parser.add_argument('--leap_size', default=-0.1, type=float)
    parser.add_argument('--arch', default=[10, 10], type=list)

    parser.add_argument('--train_samples', type=int, default=200,
                        help="""Number of samples used in batches to fit the 
                        variational approximation""")

    parser.add_argument("--data_dim", type=int, default=2)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=25000,
                        help="Number of samples to draw")

    parser.add_argument('--num_chains', type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=-1e-3,
                        help="""initial learning rate. If negative then rate "
                              is randomly chosen in range [10^-7, 10^-2.5]""")

    parser.add_argument("--train_plot", default=True, type=bool)
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()
    main(args)