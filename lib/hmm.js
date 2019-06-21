
const tf = require('@tensorflow/tfjs-core');
const seedrandom = require('seedrandom');
const Gaussian = require('./gaussian.js');
const Initialize = require('./initialize.js');

//
// The notation and equations in this module are based on Bishop:
//      Christopher M. Bishop, Pattern Recognition and Machine Learning
//      https://dl.acm.org/citation.cfm?id=1162264
// Additionally some of the Viterbi algorithm, is from Wikipedia:
//      https://en.wikipedia.org/wiki/Viterbi_algorithm
//

function zerodiv (a, b) {
  const unsafeDiv = tf.div(a, b);
  return tf.where(
    tf.equal(unsafeDiv, unsafeDiv),
    unsafeDiv,
    tf.onesLike(unsafeDiv)
  );
}

function randomUniform (shape, dtype, seed) {
  // This is how randomNormal handles the seed input
  // SEE: https://github.com/tensorflow/tfjs-core/blob/v0.12.0/src/ops/rand.ts
  const seedValue = seed || Math.random();
  const random = seedrandom.alea(seedValue.toString());
  return tf.rand(shape, random, dtype);
}

class HMM {
  constructor ({ states, dimensions, seed }) {
    if (typeof states !== 'number' ||
        Math.floor(states) !== states ||
        states <= 0) {
      throw new TypeError('states must be an integer >= 1');
    }

    if (typeof dimensions !== 'number' ||
        Math.floor(dimensions) !== dimensions ||
        dimensions <= 0) {
      throw new TypeError('dimensions must be an integer >= 1');
    }

    this._states = states;
    this._dimensions = dimensions;

    // Prior properbility (\pi) and transfer properbility (A)
    this._pi = tf.variable(
      tf.zeros([this._states]),
      false, null, 'float32'
    );

    this._A = tf.variable(
      tf.zeros([this._states, this._states]),
      false, null, 'float32'
    );

    // mean (\mu) and covariance (\Sigma) for the Gaussian emissions
    this._mu = tf.variable(
      tf.zeros([this._states, this._dimensions], 'float32'),
      false, null, 'float32'
    );
    this._Sigma = tf.variable(
      tf.zeros([this._states, this._dimensions, this._dimensions], 'float32'),
      false, null, 'float32'
    );

    this._gaussian = new Gaussian({
      mu: this._mu,
      Sigma: this._Sigma
    });
    this._initialize = new Initialize({
      states: this._states,
      dimensions: this._dimensions
    });
  }

  async setParameters ({ pi, A, mu, Sigma }) {
    tf.tidy(() => {
      // Update parameters
      this._pi.assign(pi);
      this._A.assign(A);
      this._mu.assign(mu);
      this._Sigma.assign(Sigma);
    });

    await this._gaussian.update(true);
  }

  getParameters () {
    return {
      pi: this._pi,
      A: this._A,
      mu: this._mu,
      Sigma: this._Sigma
    };
  }

  _forward (pdf) {
    // NOTE: Bishop (13.59) – Forward pass, compute \alpha, and c
    const p_x_t_list = tf.unstack(pdf); // [(N, S)]

    // Initialization
    // NOTE: Bishop (13.37)
    const p_x0 = p_x_t_list[0]; // (N, S)
    const alpha_0 = tf.mul(tf.expandDims(this._pi, 0), p_x0); // (N, S)
    // NOTE: Bishop (p. 628) – Compute c_0, as the normalization factor
    const c_0 = tf.sum(alpha_0, 1); // (N, )
    // NOTE: Bishop (13.58) - Convert alpha_0 to alpha_hat_0
    const alpha_hat_0 = tf.div(alpha_0, tf.expandDims(c_0, 1)); // (N, S)

    // Prepear recursion over time
    const alpha_hat_list = [alpha_hat_0]; // [(N, S)]
    const c_list = [c_0]; // [(N,)]

    // Perform forward recursion over time
    for (let t = 1; t < pdf.shape[0]; t++) {
      const alpha_hat_tm1 = alpha_hat_list[t - 1]; // (N, S)
      // const c_tm1 = c_list[t - 1]; // (N, )
      const p_x_t = p_x_t_list[t]; // (N, S)

      // NOTE: Bishop (13.59)
      // \sum_{z_{n-1}} \hat{\alpha(z_{n-1})} p(z_n, z_{n-1})
      const c_alpha_hat_t_sum = tf.matMul(alpha_hat_tm1, this._A); // (N, S)
      // p(x_n|z_n) \sum_{z_{n-1}} \hat{\alpha(z_{n-1})} p(z_n, z_{n-1})
      const c_alpha_hat_t = tf.mul(p_x_t, c_alpha_hat_t_sum); // (N, S)

      // NOTE: Bishop (p. 628) – Compute c_t, as the normalization factor
      const c_t = tf.sum(c_alpha_hat_t, 1); // (N, )
      const alpha_hat_t = tf.div(c_alpha_hat_t, tf.expandDims(c_t, 1)); // (N, S)

      alpha_hat_list.push(alpha_hat_t);
      c_list.push(c_t);
    }

    return [
      tf.stack(alpha_hat_list), // (N, T, S)
      tf.stack(c_list) // (N, T)
    ];
  }

  _backward (pdf, c) {
    // NOTE: Bishop (13.62) – Backward pass: compute \beta

    // NOTE: Bishop (p. 622) - Initialize as 1-vector
    const beta_T = tf.ones([pdf.shape[1], this._states]); // (N, S)

    // Perform forward recursion over time
    const beta_list = [beta_T]; // [(N, S)]
    const p_x_t_list = tf.unstack(pdf); // [(N, S)]
    const c_t_list = tf.unstack(c); // [(N, )]

    for (let t = pdf.shape[0] - 2; t >= 0; t--) {
      const beta_tp1 = beta_list[0]; // (N, S)
      const p_x_tp1 = p_x_t_list[t + 1]; // (N, S)
      const c_tp1 = c_t_list[t + 1]; // (N,)

      // NOTE: Bishop (13.62)
      // \hat{\beta}(z_{n+1}) p(x_{n+1}|z_{n+1})
      const beta_tp1_p_x_tp1 = tf.mul(beta_tp1, p_x_tp1); // (N, S)
      // \sum_{z_{n+1}} \hat{\beta}(z_{n+1}) p(x_{n+1}|z_{n+1}) p(z_{n+1}|z_n)
      const c_beta_t = tf.matMul(beta_tp1_p_x_tp1, this._A, false, true); // (N, S)

      // Normalize with c_tp1
      const beta_t = tf.div(c_beta_t, tf.expandDims(c_tp1, 1)); // (N, S)

      beta_list.unshift(beta_t);
    }

    return tf.stack(beta_list); // (T, N, S)
  }

  _expectation (pdf) {
    // Expectation: compute \gamma, and \xi
    // NOTE: Bishop (sec. 13.2.4) – This implements the numerically stable
    // using "scaling factors".

    // Perform the Forward-Backward algorithm
    const [alpha_hat, c] = this._forward(pdf); // alpha: (T, N, S), c: (T, N)
    const beta_hat = this._backward(pdf, c); // beta: (T, N, S)

    // NOTE: Bishop (13.64) – Compute \gamma
    const gamma = tf.mul(alpha_hat, beta_hat); // (T, N, S)

    // NOTE: Bishop (13.65) – Compute \xi
    // Pad c, \alpha, p(x_n|z_n), \beta, such that alpha is offset
    const c_pad = tf.pad(c, [[0, 1], [0, 0]]);
    const alpha_hat_pad = tf.pad(alpha_hat, [[1, 0], [0, 0], [0, 0]]);
    const p_x_pad = tf.pad(pdf, [[0, 1], [0, 0], [0, 0]]);
    const beta_hat_pad = tf.pad(beta_hat, [[0, 1], [0, 0], [0, 0]]);

    // c_n^{-1} \hat{\alpha} p(x_n|z_n) p(z_n|z_{n-1}) \hat{\beta}(z_n)
    const c_inv_alpha = tf.div(
      alpha_hat_pad, // (T+1, N, S)
      tf.expandDims(c_pad, -1) // (T+1, N, 1)
    );
    const c_inv_alpha_p_x = tf.mul(
      tf.expandDims(c_inv_alpha, -1), // (T+1, N, S, 1)
      tf.expandDims(p_x_pad, -2) // (T+1, N, 1, S)
    );
    const c_inv_alpha_p_x_A = tf.mul(
      c_inv_alpha_p_x, // (T+1, N, S, S)
      tf.expandDims(tf.expandDims(this._A, 0), 0) // (1, 1, S, S)
    );
    const xi_pad = tf.mul(
      c_inv_alpha_p_x_A, // (T+1, N, S, S)
      tf.expandDims(beta_hat_pad, -2) // (T+1, N, 1, S)
    );

    // Strip padding from \xi
    const T = pdf.shape[0];
    const xi = tf.slice(xi_pad, [1, 0, 0, 0], [T - 1, -1, -1, -1]); // (T, N, S, S)

    return [gamma, xi];
  }

  _maximizationPi (gamma) {
    // NOTE: Bishop (13.18) – Maximization: compute \pi
    const gamma_0 = tf.reshape(
      tf.gather(gamma, [0]),
      [gamma.shape[1], this._states]
    ); // (N, S)
    const gamma_0_mv = tf.mean(gamma_0, 0); // (S, )
    // TODO: Unsure why reference, has no normalization of pi
    const pi = zerodiv(
      gamma_0_mv,
      tf.sum(gamma_0_mv, 0, true) // (1, )
    ); // (S, )
    return pi;
  }

  _maximizationA (xi) {
    // NOTE: Bishop (13.19) – Maximization: compute A
    const xi_mv = tf.mean(xi, 1); // (T, S, S)
    const sum_xi_mv = tf.sum(xi_mv, 0); // (S, S)
    const A = zerodiv(
      sum_xi_mv,
      tf.sum(sum_xi_mv, 1, true) // (S, 1)
    );
    return A;
  }

  _maximizationMu (data /* (T, N, D) */, gamma /* (T, N, S) */) {
    // NOTE: Bishop (13.20) – Maximization: compute \mu
    const gamma_x = tf.mul(
      tf.expandDims(gamma, -1), // (T, N, S, 1)
      tf.expandDims(data, -2) // (T, N, 1, D)
    ); // (T, N, S, D)
    const sum_gamma_x = tf.sum(gamma_x, 0); // (N, S, D)

    // NOTE: The outer tf.sum is really a tf.mean, but since they both mean
    // over N, the factor cancels in the division.
    // It is not entirely clear why doing a mean before the division is the
    // correct choice. However, it does prevent some NaN issues where
    // gamma == 0. And summing over N, makes \sum_N \sum_T gamma_{n,t} less
    // likely to be zero.
    const mu = zerodiv(
      tf.sum(sum_gamma_x, 0), // (S, D)
      tf.expandDims(tf.sum(tf.sum(gamma, 0), 0), -1) // (S, 1)
    ); // (S, D)
    return mu;
  }

  _maximizationSigma (data /* (T, N, D) */, gamma /* (T, N, S) */) {
    // NOTE: Bishop (13.21) – Maximization: compute \Sigma
    const x_m_mu = tf.sub(
      tf.expandDims(data, 2), // (T, N, 1, D)
      tf.expandDims(tf.expandDims(this._mu, 0), 0) // (1, 1, S, D)
    ); // (T, N, S, D)
    const x_m_mu_2 = tf.mul(
      tf.expandDims(x_m_mu, -1), // (T, N, S, D, 1),
      tf.expandDims(x_m_mu, -2) // (T, N, S, 1, D)
    ); // (T, N, S, D, D)
    const gamma_x_m_mu_2 = tf.mul(
      tf.expandDims(tf.expandDims(gamma, -1), -1), // (T, N, S, 1, 1)
      x_m_mu_2 // (T, N, S, D, D)
    );
    const sum_gamma_x_m_mu_2 = tf.sum(gamma_x_m_mu_2, 0); // (N, S, D, D)

    // NOTE: The outer tf.sum is really a tf.mean, but since they both mean
    // over N, the factor cancels in the division.
    // It is not entirely clear why doing a mean before the division is the
    // correct choice. However, it does prevent some NaN issues where
    // gamma == 0. And summing over N, makes \sum_N \sum_T gamma_{n,t} less
    // likely to be zero.
    const Sigma = zerodiv(
      tf.sum(sum_gamma_x_m_mu_2, 0), // (S, D, D)
      tf.expandDims(tf.expandDims(tf.sum(tf.sum(gamma, 0), 0), -1), -1) // (S, 1, 1)
    ); // (S, D, D)
    return Sigma;
  }

  _maximization (data, gamma, xi) {
    // Expectation: compute \pi, A, \mu, and \Sigma
    const pi = this._maximizationPi(gamma);
    const A = this._maximizationA(xi);
    const mu = this._maximizationMu(data, gamma);
    const Sigma = this._maximizationSigma(data, gamma);

    return [pi, A, mu, Sigma];
  }

  async fit (data, { maxIterations = 100, tolerance = 0.001, seed = undefined } = {}) {
    // data should have the shape (observations, time, dimensions) = (N, T, D)

    // Swap first two axis, such time is first. This makes recursion over
    // time much easier.
    data = tf.tidy(`hmm-fit-pre`, () => tf.transpose(data, [1, 0, 2])); // (T, N, D)

    // NOTE: Bishop (p. 623) - Use KMeans to initialize \mu and \Sigma
    const [muInit, SigmaInit] = await this._initialize.compute(data, {
      maxIterations, tolerance, seed
    });
    this._mu.assign(muInit); // (S, D)
    this._Sigma.assign(SigmaInit); // (S, D, D)

    // NOTE: Bishop (p. 618) - Intialize evenly. Bishop (p. 617), also
    // suggests initializing randomly, however evenly is much easier.
    this._A.assign(
      tf.fill([this._states, this._states], 1 / this._states)
    );
    this._pi.assign(
      tf.fill([this._states], 1 / this._states)
    );

    // Iterate the EM-algorithm
    let converged = false;
    let iteration = 0;
    let maxDiff;
    for (; iteration < maxIterations; iteration++) {
      // Update gaussian internals now that \Sigma has been updated
      await this._gaussian.update();

      maxDiff = await tf.tidy(`hmm-fit-${iteration}`, () => {
        // Precompute pdf for the data
        const pdf = this._gaussian.pdf(data); // (T, N, S)

        // Perform EM-step
        // gamma: (T, N, S), xi: (T, N, S, S)
        const [gamma, xi] = this._expectation(pdf);
        // pi: (S), A: (S, S), mu: (D), Sigma: (D, D)
        const [pi, A, mu, Sigma] = this._maximization(data, gamma, xi);

        // Compute difference
        const piDiff = tf.max(tf.abs(tf.sub(pi, this._pi)));
        const ADiff = tf.max(tf.abs(tf.sub(A, this._A)));
        const muDiff = tf.max(tf.abs(tf.sub(mu, this._mu)));
        const SigmaDiff = tf.max(tf.abs(tf.sub(Sigma, this._Sigma)));

        // NOTE: Bishop suggests this as the convergence critieria.
        // While R's em.control {depmixS4}, uses
        //   (log L(i) - log L(i-1))/(log L(i-1)) < tol.
        // At the very least, consider using a relative diff instead of an
        // absolute.
        const maxDiff = tf.max(tf.stack([ piDiff, ADiff, muDiff, SigmaDiff ]));

        // Update parameters
        this._pi.assign(pi);
        this._A.assign(A);
        this._mu.assign(mu);
        this._Sigma.assign(Sigma);

        return maxDiff;
      }).data();

      // Check if converged
      if (maxDiff < tolerance) {
        converged = true;
        break;
      }
    }

    // Finalize gaussian internals now that \Sigma has been updated
    await this._gaussian.update(true);

    // Done
    return {
      iterations: iteration + 1,
      converged: converged,
      tolerance: maxDiff
    };
  }

  logLikelihood (data) {
    // data should have the shape (observations, time, dimensions) = (N, T, D)

    return tf.tidy('hmm-log-likelihood', () => {
      // Swap first two axis, such time is first. This makes recursion over
      // time much easier.
      data = tf.transpose(data, [1, 0, 2]);

      // Precompute pdf for the data
      const pdf = this._gaussian.pdf(data); // (T, N, S)

      // alpha: (T, N, S), c: (T, N)
      const [, c] = this._forward(pdf);
      const log_likelihood = tf.sum(tf.log(c), 0); // (N, )

      return log_likelihood;
    });
  }

  _viterbi_forward (pdf) {
    const p_x_t_list = tf.unstack(pdf); // [(N, S)]

    // NOTE: Bishop (13.69) - viterbi initialization
    const w_0 = tf.add(
      tf.expandDims(tf.log(this._pi), 0), // (1, S)
      tf.log(p_x_t_list[0]) // (N, S)
    ); // (N, S)

    // NOTE: Bishop (13.68) - recursion
    const w_list = [w_0];
    const argmax_w_list = [tf.zeros(w_0.shape, 'int32')];
    for (let t = 1; t < pdf.shape[0]; t++) {
      const w_tm1 = w_list[t - 1]; // (N, S)

      const objective = tf.add(
        tf.expandDims(tf.log(this._A), 0), // (1, S, S)
        tf.expandDims(w_tm1, -1) // (N, S, 1)
      ); // (N, S, S)

      const w_t = tf.add(
        tf.log(p_x_t_list[t]), // (N, S)
        tf.max(objective, 1) // (N, S)
      ); // (N, S)
      const argmax_w_t = tf.argMax(objective, 1); // (N, S)

      w_list.push(w_t);
      argmax_w_list.push(argmax_w_t);
    }

    return [
      tf.stack(w_list), // (T, N, S)
      tf.stack(argmax_w_list) // (T, N, S)
    ];
  }

  _viterbi_backward (w, argmax_w) {
    // NOTE: See https://en.wikipedia.org/wiki/Viterbi_algorithm
    // Alternatively, see Bishop (13.71) but this is less explicit.
    const T = w.shape[0];
    const N = w.shape[1];

    // Initialization
    const w_T = tf.reshape(
      tf.slice(w, [T - 1, 0, 0], [1, -1, -1]), // (1, N, S)
      [N, this._states]
    ); // (N, S)
    const state_T = tf.argMax(w_T, 1); // (N, )

    // Recursion
    const state_list = [state_T]; // [(N, )]
    const argmax_w_list = tf.unstack(argmax_w);
    for (let t = T - 2; t >= 0; t--) {
      const state_tp1 = state_list[0]; // (N, )
      const argmax_w_tp1 = argmax_w_list[t + 1]; // (N, S)

      // Tensorflow.JS does not have advanced indexing, so use tf.gather
      // by flattning the input and transform path_tp1 into strided indices.
      const state_t = tf.gather(
        tf.reshape(argmax_w_tp1, [N * this._states]), // (N * S, )
        tf.add(
          tf.range(0, N * this._states, this._states, 'int32'), // [0, 3, 6...]
          state_tp1
        ) // (N, )
      ); // (N, )

      state_list.unshift(state_t);
    }

    return tf.stack(state_list); // (T, N)
  }

  inference (data) {
    // data should have the shape (observations, time, dimensions) = (N, T, D)

    return tf.tidy('hmm-inference', () => {
      // Swap first two axis, such time is first. This makes recursion over
      // time much easier.
      data = tf.transpose(data, [1, 0, 2]);

      // Precompute pdf for the data
      const pdf = this._gaussian.pdf(data); // (T, N, S)

      // w: (T, N, S), argmax_w: (T, N, S)
      const [w, argmax_w] = this._viterbi_forward(pdf);
      // state: (T, N)
      const state = this._viterbi_backward(w, argmax_w);

      return tf.transpose(state); // (N, T)
    });
  }

  _sample_choose (uniform /* (N, ) */, cumsum /* (1 or N, S) */) {
    const uniform_threshold = tf.cast(tf.less(
      tf.expandDims(uniform, -1), // (N, 1)
      cumsum // (1 or N, S)
    ), 'int32'); // (N, S)

    // Details on sampling from choose / histogram:
    // Consider the distribution
    // [0.2, 0.3, 0.499]
    // That has the cumsum:
    // [0.2, 0.5, 0.99]
    // The the sum of the threshold will be:
    // sample 0, [True, True, True] = 3
    // sample 0.9: [False, False, True] = 1
    // sample 0.999999: [False, False, False] = 0
    // Use max((S - sum), S - 1) to compute the index using cumsum
    const choose_unsafe = tf.sub(
      tf.tensor([cumsum.shape[1]], [1], 'int32'), // (1, )
      tf.sum(uniform_threshold, 1) // (N, )
    );
    const choose = tf.minimum(
      choose_unsafe, // (N, )
      tf.tensor([cumsum.shape[1] - 1], [1], 'int32') // (1, )
    ); // (N, )
    return choose;
  }

  sample ({ observations, time, seed = undefined }) {
    let seedCounter = 0;
    function nextSeed () {
      return typeof seed === 'number' ? seed + (++seedCounter) : seed;
    }

    return tf.tidy('hmm-sample', () => {
      // Prepear cumsum for sampling
      const cumsum_pi = tf.cumsum(this._pi); // (S, )
      const cumsum_A = tf.cumsum(this._A, 1); // (S, S)

      // Compute uniforms in one go
      const uniform = randomUniform(
        [time, observations], 'float32', nextSeed()
      );
      const uniform_list = tf.unstack(uniform); // [(N, )]

      // Initialization
      const uniform_0 = uniform_list[0]; // (N, )
      const states_0 = this._sample_choose(
        uniform_0,
        tf.expandDims(cumsum_pi, 0)
      );
      const emissions_0 = this._gaussian.sample(states_0, {
        seed: nextSeed()
      });

      // Recursion
      const states_list = [states_0];
      const emissions_list = [emissions_0];
      for (let t = 1; t < time; t++) {
        const states_tm1 = states_list[t - 1];
        const uniform_t = uniform_list[t];

        // Sample state and emissions
        const states_t = this._sample_choose(
          uniform_t,
          tf.gather(cumsum_A, states_tm1)
        );
        const emissions_t = this._gaussian.sample(states_t, {
          seed: nextSeed()
        });

        states_list.push(states_t);
        emissions_list.push(emissions_t);
      }

      // Done
      return {
        states: tf.transpose(tf.stack(states_list), [1, 0]),
        emissions: tf.transpose(tf.stack(emissions_list), [1, 0, 2])
      };
    });
  }
}

module.exports = HMM;
