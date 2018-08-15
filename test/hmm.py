"""Modified and Simplified version of https://github.com/kesmarag/ml-hmm

Copyright (c) 2018 Costas Smaragdakis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans


class HMM(object):
    """A Hidden Markov Models class with Gaussians emission distributions.
    """

    def __init__(self, num_states, data_dim, obs=None, time=None):
        """Init method for the HiddenMarkovModel class.

        Args:
          num_states: Number of states.
          data_dim: Dimensionality of the observed data.
        """
        self._num_obs = obs
        self._num_time = time

        self._graph = tf.Graph()
        self._num_states = num_states
        self._data_dim = data_dim
        self._em_probs = self._emission_probs_family()
        # numpy variables
        self._p0, self._tp = self._init_p0_tp()
        self._mu = np.random.rand(self._num_states, self._data_dim)
        self._sigma = np.array(
            [np.identity(self._data_dim, dtype=np.float64)] * self._num_states
        )
        # creation of the computation graph
        self._create_the_computational_graph()

    def posterior(self, data):
        """Runs the forward-backward algorithm in order to calculate
           the log-scale posterior probabilities.

        Args:
          data: A numpy array with rank two or three.

        Returns:
          A numpy array that contains the log-scale posterior probabilities of
          each time serie in data.

        """
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self._dataset_tf: data, self._p0_tf: self._p0,
                self._tp_tf: self._tp, self._mu_tf: self._mu,
                self._sigma_tf: self._sigma}
            return np.squeeze(sess.run(self._posterior, feed_dict=feed_dict))

    def fit(self, data, max_steps=100, TOL=0.0001):
        """Implements the Baum-Welch algorithm.

        Args:
          data: A numpy array with rank two or three.
          max_steps: The maximum number of steps.
          TOL: The tolerance for stoping training process.

        Returns:
          True if converged, False otherwise.

        """
        # Bishop, page 623 - Use KMeans to initialize \mu
        # In Bishop, they say to initialize \Sigma as the covariance matrix,
        # however here it is the identity matrix.
        data_timeless = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        kmeans = KMeans(n_clusters=self._num_states,
                        max_iter=max_steps, tol=TOL)
        states = kmeans.fit_predict(data_timeless)

        # get \mu and \Sigma
        self._mu = kmeans.cluster_centers_
        self._sigma = np.stack([
            np.cov(data_timeless[states == s], rowvar=False)
            for s in range(self._num_states)
        ])

        # NOTE: p0: prior properbility (\pi)
        # NOTE: tp: transfer properbility (A)
        # Initializes as uniform discrete distribution
        self._p0, self._tp = self._init_p0_tp()

        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(max_steps):
                feed_dict = {
                    self._dataset_tf: data,
                    self._p0_tf: self._p0,
                    self._tp_tf: self._tp,
                    self._mu_tf: self._mu,
                    self._sigma_tf: self._sigma
                }

                if step == 0:
                    p0_prev = np.zeros((self._num_states,))
                    tp_prev = np.zeros((self._num_states, self._num_states))
                    mu_prev = np.zeros((self._num_states, self._data_dim))
                    sigma_prev = np.zeros((self._num_states, self._data_dim, self._data_dim))
                else:
                    p0_prev = self._p0
                    tp_prev = self._tp
                    mu_prev = self._mu
                    sigma_prev = self._sigma

                # Perform EM iteration
                self._p0, self._tp, self._mu, self._sigma = sess.run(
                    [self._p0_tf_new, self._tp_tf_new,
                     self._mu_tf_new, self._sigma_tf_new],
                    feed_dict=feed_dict)

                ch_p0 = np.max(np.abs(self._p0 - p0_prev))
                ch_tp = np.max(np.abs(self._tp - tp_prev))
                ch_mu = np.max(np.abs(self._mu - mu_prev))
                ch_sigma = np.max(np.abs(self._sigma - sigma_prev))
                if ch_p0 < TOL and ch_tp < TOL and ch_mu < TOL and ch_sigma < TOL:
                    converged = True
                    break

        return (converged, step)

    def run_viterbi(self, data):
        """Implements the viterbi algorithm.
        (I am not sure that it works properly)

        Args:
          data: A numpy array with rank two or three.

        Returns:
          The most probable state path.

        """
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = feed_dict = {
                self._dataset_tf: data,
                self._p0_tf: self._p0,
                self._tp_tf: self._tp,
                self._mu_tf: self._mu,
                self._sigma_tf: self._sigma
            }

            w, am = sess.run([self._w, self._am], feed_dict=feed_dict)
            w = (w[:, -1, :])

            # NOTE: Refer to https://en.wikipedia.org/wiki/Viterbi_algorithm
            # for this part.
            paths = []
            for w_obs, am_obs in zip(w, am):
                path = np.zeros((am_obs.shape[0], ), dtype=np.int32)
                paths.append(path)

                T = am_obs.shape[0]
                path[T - 1] = np.argmax(w_obs)
                for i in range(T - 1, 0, -1):
                    path[i - 1] = am_obs[i, path[i]]

            return np.asarray(paths, dtype='int16')

    def generate(self, num_samples):
        """Generate simulated data from the model.

        Args:
          num_samples: The number of samples of the generated data.

        Returns:
          The generated data.

        """
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self._p0_tf: self._p0,
                self._tp_tf: self._tp, self._mu_tf: self._mu,
                self._sigma_tf: self._sigma, self._num_samples_tf: num_samples}
            states, samples = sess.run(
                [self._states, self._samples], feed_dict=feed_dict)
            return samples, states

    @property
    def p0(self):
        return np.squeeze(self._p0)

    @property
    def tp(self):
        return self._tp

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    def _create_the_computational_graph(self):
        with self._graph.as_default():
            self._dataset_tf = tf.placeholder(
                'float64', shape=[self._num_obs, self._num_time, self._data_dim])
            self._num_samples_tf = tf.placeholder('int32')
            self._min_var_tf = tf.placeholder('float64')
            self._p0_tf = tf.placeholder(
                tf.float64, shape=[1, self._num_states])
            self._tp_tf = tf.placeholder(
                tf.float64, shape=[self._num_states, self._num_states])
            self._emissions_eval()
            self._forward()
            self._backward()
            self._expectation()
            self._maximization()
            self._simulate()
            self._viterbi()

    def _emission_probs_family(self):
        with self._graph.as_default():
            self._mu_tf = tf.placeholder(
                tf.float64, shape=[self._num_states, self._data_dim]
            )
            self._sigma_tf = tf.placeholder(
                tf.float64,
                shape=[self._num_states, self._data_dim, self._data_dim]
            )
            return tfp.distributions.MultivariateNormalFullCovariance(
                loc=self._mu_tf,
                covariance_matrix=self._sigma_tf
            )

    def _emissions_eval(self):
        with tf.variable_scope('emissions_eval'):
            dataset_expanded = tf.expand_dims(self._dataset_tf, -2)
            self._emissions = self._em_probs.prob(dataset_expanded)

    def _forward_step(self, n, alpha, c):
        # calculate alpha[n-1] tp
        alpha_tp = tf.matmul(alpha[n - 1], self._tp_tf)
        # calculate p(x|z) \sum_z alpha[n-1] tp
        a_n_tmp = tf.multiply(tf.squeeze(self._emissions[:, n, :]), alpha_tp)
        c_n_tmp = tf.expand_dims(tf.reduce_sum(a_n_tmp, axis=-1), -1)
        return [
            n + 1,
            tf.concat([alpha, tf.expand_dims(a_n_tmp / c_n_tmp, 0)], 0),
            tf.concat([c, tf.expand_dims(c_n_tmp, 0)], 0)
        ]

    def _forward(self):
        with tf.variable_scope('forward'):
            n = tf.shape(self._dataset_tf)[1]
            # alpha shape : (N, I, states)
            # c shape : (N, I, 1)
            # NOTE: Bishop (13.37)
            a_0_tmp = tf.expand_dims(
                tf.multiply(
                    self._emissions[:, 0, :],
                    tf.squeeze(self._p0_tf)), 0)
            # NOTE: Compute c_0, as the normalization factor
            c_0 = tf.expand_dims(tf.reduce_sum(a_0_tmp, axis=-1), -1)
            alpha_0 = a_0_tmp / c_0

            # NOTE: Bishop (13.59), again c is computed as the normalization
            # factor
            i0 = tf.constant(1)
            condition_forward = lambda i, alpha, c: tf.less(i, n)
            _, self._alpha, self._c = tf.while_loop(
                condition_forward,
                self._forward_step,
                [i0, alpha_0, c_0],
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape(
                        [None, alpha_0.shape[1], self._num_states]),
                    tf.TensorShape([None, c_0.shape[1], 1])
                ]
            )

            # NOTE: Bishop (13.63), but in log space
            self._posterior = tf.reduce_sum(tf.log(self._c), axis=0)

    def _backward_step(self, n, betta, b_p):
        b_p_tmp = tf.multiply(betta[0], tf.squeeze(self._emissions[:, -n, :]))
        b_n_tmp = tf.matmul(b_p_tmp, self._tp_tf,
                            transpose_b=True) / self._c[-n]
        return [
            n + 1,
            tf.concat([tf.expand_dims(b_n_tmp, 0), betta], 0),
            tf.concat([tf.expand_dims(b_p_tmp, 0), b_p], 0)
        ]

    def _backward(self):
        with tf.variable_scope('backward'):
            n = tf.shape(self._dataset_tf)[1]
            shape = tf.shape(self._dataset_tf)[0]
            dims = tf.stack([shape, self._num_states])

            b_tmp_ = tf.fill(dims, 1.0)
            betta_0 = tf.expand_dims(tf.ones_like(b_tmp_, dtype=tf.float64), 0)
            b_p_0 = tf.expand_dims(tf.ones_like(b_tmp_, dtype=tf.float64), 0)

            # NOTE: Bishop (13.62)
            i0 = tf.constant(1)
            condition_backward = lambda i, betta, b_p: tf.less(i, n)
            _, self._betta, b_p_tmp = tf.while_loop(
                condition_backward,
                self._backward_step,
                [i0, betta_0, b_p_0],
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape([None, None, self._num_states]),
                    tf.TensorShape([None, None, self._num_states])
                ]
            )
            self._b_p = b_p_tmp[:-1, :, :]

    def _simulate_step(self, n, states, samples):
        state = tf.expand_dims(
            tf.where(
                tf.squeeze(
                    self._cum_tp_tf[
                        tf.cast(states[n - 1, 0], dtype='int32')
                    ] > self._rand[n]))[0],
            0)
        sample = tf.expand_dims(self._em_probs.sample()[tf.cast(
            state[0, 0], dtype='int32')], 0)
        return [n + 1, tf.concat(
            [states, state], 0), tf.concat([samples, sample], 0)]

    def _simulate(self):
        with self._graph.as_default():
            self._rand = tf.random_uniform(
                [self._num_samples_tf, 1], maxval=1.0, dtype='float64')
            self._cum_p0_tf = tf.cumsum(self._p0_tf, axis=1)
            self._cum_tp_tf = tf.cumsum(self._tp_tf, axis=1)
            # initial sample
            _init_sample_state = tf.expand_dims(
                tf.where(tf.squeeze(self._cum_p0_tf > self._rand[0]))[0], 0)
            _init_sample = tf.expand_dims(self._em_probs.sample()[tf.cast(
                _init_sample_state[0, 0], dtype='int32')], 0)

            i0 = tf.constant(1, dtype='int32')
            condition_sim = lambda i, states, samples: tf.less(
                i, self._num_samples_tf)
            _, self._states, self._samples = tf.while_loop(
                condition_sim, self._simulate_step,
                [i0, _init_sample_state, _init_sample],
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape([None, 1]),
                    tf.TensorShape([None, self._data_dim])
                ])

    def _xi_calc(self, n, xi):
        a_c = tf.expand_dims(self._alpha[n - 1] / self._c[n], -1)
        b_p = tf.expand_dims(self._b_p[n - 1], -1)
        a_b_p = tf.matmul(a_c, b_p, transpose_b=True)
        xi_n_tmp = tf.multiply(a_b_p, self._tp_tf)
        return [n + 1, tf.concat([xi, tf.expand_dims(xi_n_tmp, 0)], 0)]

    def _expectation(self):
        with tf.variable_scope('expectation'):
            # gamma shape : (N, I, states)
            # NOTE: Bishop (13.64), numerically stable version
            self._gamma = tf.multiply(self._alpha, self._betta, name='gamma')

            # NOTE: Bishop (13.65), note that the \beta part is hidden
            # in the following variable.
            # _b_p: \beta(z_n) * p(x_n|z_n)
            n = tf.shape(self._dataset_tf)[1]
            shape = tf.shape(self._dataset_tf)[0]
            dims = tf.stack([shape, self._num_states, self._num_states])

            xi_tmp_ = tf.fill(dims, 1.0)
            xi_0 = tf.expand_dims(tf.ones_like(xi_tmp_, dtype=tf.float64), 0)
            i0 = tf.constant(1)
            condition_xi = lambda i, xi: tf.less(i, n)
            _, xi_tmp = tf.while_loop(
                condition_xi, self._xi_calc, [i0, xi_0],
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape([None, None, self._num_states, self._num_states])
                ])

            self._xi = xi_tmp[1:, :, :]

    def _maximization(self):
        with tf.variable_scope('maximization'):
            max_var = 20.0

            # update the initial state probabilities
            # NOTE: Bishop (13.18)
            gamma_mv = tf.reduce_mean(self._gamma, axis=1, name='gamma_mv')
            self._p0_tf_new = tf.transpose(tf.expand_dims(gamma_mv[0], -1))

            # update the transition matrix
            # first calculate sum_n=2^{N} xi_mean[n-1,k , n,l]
            # NOTE: Bishop (13.19)
            xi_mv = tf.reduce_mean(self._xi, axis=1, name='xi_mv')
            sum_xi_mean = tf.squeeze(tf.reduce_sum(xi_mv, axis=0))
            self._tp_tf_new = sum_xi_mean / (tf.reduce_sum(sum_xi_mean,
                                                           axis=1,
                                                           keepdims=True))

            # mu update
            # NOTE: Bishop (13.20)
            x_t = tf.transpose(self._dataset_tf, perm=[1, 0, 2],
                               name='x_transpose')
            gamma_x = tf.matmul(tf.expand_dims(self._gamma, -1),
                                tf.expand_dims(x_t, -1), transpose_b=True)
            sum_gamma_x = tf.reduce_sum(gamma_x, axis=[0, 1])
            mu_tmp_t = tf.transpose(sum_gamma_x) / tf.reduce_sum(
                self._gamma,
                axis=[0, 1])
            self._mu_tf_new = tf.transpose(mu_tmp_t)

            # update the covariances
            # NOTE: Bishop (13.21)
            # gamma shape : (N, I, states)
            # x shape : (I, N, dim)
            # mu shape : (states, dim)
            x_expanded = tf.expand_dims(self._dataset_tf, -2)
            # calculate (x - mu) tensor : expected shape (I, N, states, dim)
            x_m_mu = tf.subtract(x_expanded, self._mu_tf)
            # calculate (x - mu)(x - mu)^T : expected shape (I, N, states, dim,
            # dim)
            x_m_mu_2 = tf.matmul(tf.expand_dims(x_m_mu, -1),
                                 tf.expand_dims(x_m_mu, -2))
            gamma_r = tf.transpose(self._gamma, perm=[1, 0, 2])
            gamma_x_m_mu_2 = tf.multiply(
                x_m_mu_2,
                tf.expand_dims(tf.expand_dims(gamma_r, -1), -1))
            _new_cov_tmp = tf.reduce_sum(
                gamma_x_m_mu_2,
                axis=[0, 1]) / tf.expand_dims(
                tf.expand_dims(
                    tf.reduce_sum(
                        gamma_r,
                        axis=[0, 1]), -1), -1)

            # NOTE: not part of the standard equeations
            # Implements a min max threshold on the covariance
            self._sigma_tf_new = _new_cov_tmp

    def _viterbi_step(self, n, w, am):
        # NOTE: Bishop (13.68)
        w_tmp = tf.expand_dims(
            tf.log(self._emissions[:, n]) +
            tf.reduce_max(
                tf.expand_dims(w[:, n - 1], -1) +
                tf.expand_dims((tf.log(self._tp_tf)), 0),
                axis=-2
            ), 1
        )

        # NOTE: argmax of Bishop (13.68)
        am_tmp = tf.expand_dims(tf.argmax(
            tf.expand_dims(w[:, n - 1], -1) +
            tf.expand_dims((tf.log(self._tp_tf)), 0),
            axis=-2
        ), 1)

        return [
            n + 1,
            tf.concat([w, w_tmp], 1),
            tf.concat([am, am_tmp], 1)
        ]

    def _viterbi(self):
        with self._graph.as_default():
            m = tf.shape(self._dataset_tf)[0]
            n = tf.shape(self._dataset_tf)[1]

            # NOTE: Bishop (13.69)
            w1 = tf.expand_dims(
                tf.log(self._p0_tf) + tf.log(self._emissions[:, 0]), 1
            )

            am1 = tf.zeros_like(w1, dtype='int64')

            i0 = tf.constant(1)
            condition_viterbi = lambda i, w, am: tf.less(i, n)
            _, self._w, self._am = tf.while_loop(
                condition_viterbi,
                self._viterbi_step,
                [i0, w1, am1],
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape([None, None, self._num_states]),
                    tf.TensorShape([None, None, self._num_states])
                ])

    def _init_p0_tp(self):
        tp = np.ones([self._num_states, self._num_states],
                     dtype=np.float64) / self._num_states
        p0 = np.ones([1, self._num_states], dtype=np.float64) / self._num_states

        return p0, tp
