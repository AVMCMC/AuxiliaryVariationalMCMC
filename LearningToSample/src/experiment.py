""" Class to manage running experiments"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from LearningToSample.src.logging import make_run_dir
from LearningToSample.src.eval import batch_means_ess, gelman_rubin_diagnostic, acceptance_rate_2



class Experiment:

    def __init__(self, log_dir, sampler, dist, params, debug=False):
        self.dist = dist
        self.logs = make_run_dir(log_dir)
        self.sess = tf.Session()
        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sampler = sampler
        self.args = params
        self._save_params(params)
        self.writer = tf.summary.FileWriter(self.logs['logs'])
        self.args['save_path'] = self.logs
        self.args['sess'] = self.sess
        self.args['writer'] = self.writer

    def run(self):

        print('starting training')
        losses, train_time = self.sampler.train(**self.args)

        np.save(os.path.join(self.logs['results'], 'losses.npy'), np.array(losses))
        self._plot(losses[1:], 'losses.png')
        self._save_logs('results.txt', 'train time', train_time)

        print('drawing samples')
        samples, sample_time = self.sampler.sample(**self.args)

        samples = samples[self.args['burn_in']:]
        np.save(os.path.join(self.logs['results'], 'samples.npy'),
                samples)
        self._save_logs('results.txt', 'sample time', sample_time)

        print('evaluating samples')
        ess = batch_means_ess(samples)
        gr = gelman_rubin_diagnostic(samples)
        acc_rate = acceptance_rate_2(samples)
        np.save(os.path.join(self.logs['results'], 'ess.npy'), ess)
        min_ess = np.mean(np.min(ess, axis=1), axis=0)
        std_ess = np.std(np.min(ess, axis=1), axis=0)
        self._save_logs('results.txt', 'min_ess', min_ess)
        self._save_logs('results.txt', 'std_ess', std_ess)
        self._save_logs('results.txt', 'num_samples:', samples.size)
        self._save_logs('results.txt', 'gelman_rubin:', gr)
        self._save_logs('results.txt', 'acceptance_rate:', acc_rate)
        self._trace_plot(samples)
        eval, eval_std = self._evaluate(samples)
        self._save_logs('results.txt', 'eval_metric', eval)
        self._save_logs('results.txt', 'eval_metric_std', eval_std)

        if samples.shape[2] == 2:
            self._plot2d(samples)

    def _trace_plot(self, samples):
        i = np.random.randint(0, samples.shape[1])
        fig, ax = plt.subplots()
        ax.plot(samples[:, i, :], alpha=0.4)
        ax.plot(samples[:, i, :], '+')
        ax.set_xlim([0, samples.shape[0]])
        ax.set_aspect('auto', 'datalim')
        plt.savefig(os.path.join(self.logs['figs'], 'trace.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot2d(self, samples):
        fig, ax = plt.subplots()
        ax.hist2d(samples[:, 0, 0], samples[:, 0, 1], bins=200)
        ax.set_aspect('equal', 'box')
        plt.savefig(os.path.join(self.logs['figs'], 'samples.png'))
        plt.close()

    def _plot(self, data, title):
        fig, ax = plt.subplots()
        ax.set_aspect('auto', 'datalim')
        ax.plot(data)
        plt.savefig(os.path.join(self.logs['figs'], title), bbox_inches='tight')
        plt.close()

    def _evaluate(self, samples):
        T, M, D = samples.shape
        eval = []
        for i in range(M):
            eval.append(self.dist.evaluate(samples[:, i, :], self.sess))
        return np.mean(eval), np.std(eval)


    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f,  indent=4)

    def _save_logs(self, filename, savestring, arg):
        with open(os.path.join(self.logs['results'], filename), 'a') as f:
            f.write(savestring + ': ' + str(arg) + '\n')
