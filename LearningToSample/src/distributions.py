""" Simple interface to all the distributions that we use for testing"""
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd


class Distribution:

    def log_prob_func(self):
        """returns a function which when called evaluates the log
        probability of the distirbution up till an additive constant"""
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def evaluate(self, samples):
        raise NotImplementedError


class DiagonalGaussian(Distribution):

    def __init__(self, mean, scale_diag):
        """ The batch shape of samples will be the broadcast shape
        of mean and sd """
        self.mean = mean
        self.sd = scale_diag
        self.dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sd)

    def log_prob_func(self):
        return lambda x: self.dist.log_prob(x)

    def sample(self, sample_shape=()):
        return self.dist.sample(sample_shape)

    def evaluate(self, *args):
        return 0.0

class MixtureOfGaussians(Distribution):

    def __init__(self, means, stds, pis):
        self.means = means
        self.stds = stds
        self.pis = pis
        self.dist = tfd.Mixture(
            cat=tfd.Categorical(probs=pis),
            components=[tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
                        for m, s in zip(means, stds)]
        )

    def log_prob_func(self):
        return lambda x: self.dist.log_prob(x)

    def sample(self, sample_shape=()):
        return self.dist.sample(sample_shape)

    def evaluate(self, *args):
        return 0.0


class Ring(Distribution):

    def __init__(self, scale=0.2):
        self.scale = scale

    def log_prob_func(self):

        def donut_density(x):
            x = tf.reshape(x, [tf.shape(x)[0], 1, tf.shape(x)[1]])
            thetas = tf.lin_space(0., 2 * np.pi, 500)
            sins = tf.sin(thetas)
            coses = tf.cos(thetas)
            means = tf.transpose(tf.stack([sins, coses]))
            means = tf.expand_dims(means, 0)
            exponent = -(1.0 / self.scale) * tf.reduce_sum((x - means) ** 2, axis=2)
            return tf.reduce_logsumexp(exponent, axis=1)

        return donut_density

    def sample(self):
        pass

    def evaluate(self, *args):
        return 0.0

class BayesLogRegPost:

    def __init__(self, X_train, y_train, X_test, y_test, prior_var):
        self.X = tf.constant(self._normalise(X_train))
        self.y = tf.constant(y_train.astype(np.float32))
        self.X_test = tf.constant(self._normalise(X_test))
        self.y_test = tf.constant(y_test.astype(np.float32))
        self.scale = prior_var
        self.data_dim = X_test.shape[1] + 1
        self.theta = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.data_dim],
                                    name='theta')
        self.acc = self._build_evaluate()

    def sample(self):
        pass

    def log_prob_func(self):

        z = 2 * self.y - 1
        X = tf.expand_dims(self.X, 0)

        def log_prob(theta):
            theta_ = tf.reshape(theta, [-1, 1, self.data_dim])
            act = tf.reduce_sum(X * theta_, 2)  # Num_chains x Num_data
            log_probs = tf.reduce_sum(
                tf.log(tf.sigmoid(act * tf.transpose(z), name='sigmoid') + 1e-10), axis=1,
                keepdims=True)
            log_prior = self._log_prior(theta)
            return tf.squeeze(log_probs + log_prior)

        return log_prob

    def _build_evaluate(self):
        X = tf.expand_dims(self.X_test, 0)
        theta_ = tf.reshape(self.theta, [-1, 1, self.data_dim])
        preds = tf.greater_equal(tf.sigmoid(tf.reduce_sum(theta_*X, axis=2), name='sigmoid'), 0.5)
        preds = tf.cast(preds, tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, self.y_test), tf.float32), axis=0)
        return acc

    def _log_prior(self, theta):
        mahalob = - 0.5 * tf.reduce_sum(theta**2, axis=1, keepdims=True)
        return mahalob/self.scale

    def _normalise(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean)/std
        data = np.concatenate([data, np.ones((data.shape[0], 1))], axis=1)
        return data.astype(np.float32)

    def evaluate(self,  samples, sess):
        return sess.run(self.acc, feed_dict={self.theta: samples})


class FeedForwardNetPost:

    def __init__(self, X_train, y_train, X_test, y_test,
                 arch, act=tf.nn.tanh, prec=0.1):
        """We expect X to have shape (N, arch[0])"""
        self.arch = arch
        self.theta_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
        self.act = act
        self.prec = prec # precision
        self.X = tf.constant(self._normalise(X_train))
        self.y = tf.constant(y_train.astype(np.float32))
        self.X_test = tf.constant(self._normalise(X_test))
        self.y_test = tf.constant(y_test.astype(np.float32))
        self.theta = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.theta_dim])
        self.rmse = self._build_evaluate()

    def log_prob_func(self):

        def log_prob(theta):
            """ theta has shape """
            h = tf.expand_dims(self.X, 0)
            h = tf.tile(h, [tf.shape(theta)[0], 1, 1])
            weights = self._unflatten(theta)
            for W in weights[:-1]:
                h = self.act(h @ W)
            mean = h @ weights[-1]
            mahalob = - 0.5 * tf.reduce_sum((self.y - mean) ** 2, axis=2)
            prior = -0.5 * tf.reduce_sum(theta ** 2, axis=1, keepdims=True)

            return tf.squeeze(tf.reduce_sum(mahalob + self.prec * prior, axis=1))

        return log_prob

    def sample(self):
        pass

    def _build_evaluate(self):
        """ theta is required to have shape (num_samples, target_dim)
            x is required to have shape N, D"""
        weights = self._unflatten(self.theta)
        h = tf.expand_dims(self.X_test, 0)
        h = tf.tile(h, [tf.shape(self.theta)[0], 1, 1])
        for W in weights[:-1]:
            h = self.act(h @ W)
        preds = h @ weights[-1]
        rmse = tf.sqrt(tf.reduce_mean((preds - self.y_test)**2, axis=0))
        return rmse

    def _unflatten(self, theta):
        """theta is assumed to have shape (num_chains, target_dim)"""
        m = tf.shape(theta)[0]  # num chains
        weights = []
        start = 0
        for i in range(len(self.arch) - 1):
            size = self.arch[i] * self.arch[i + 1]
            w = tf.reshape(theta[:, start:start + size],
                           (m, self.arch[i], self.arch[i + 1]))
            weights.append(w)
            start += size
        return weights

    def evaluate(self,  samples, sess):
        return sess.run(self.rmse, feed_dict={self.theta: samples})

    def _normalise(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean)/std
        data = np.concatenate([data, np.ones((data.shape[0], 1))], axis=1)
        return data.astype(np.float32)
