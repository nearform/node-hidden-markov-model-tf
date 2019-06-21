
const tf = require('@tensorflow/tfjs-core');
const ndarray = require('ndarray');
const ndarrayDet = require('ndarray-determinant');
const ndarrayInv = require('ndarray-inv');
const ndarrayCholesky = require('ndarray-cholesky-factorization');

//
// The notation and equations in this module are based on Bishop:
//      Christopher M. Bishop, Pattern Recognition and Machine Learning
//      https://dl.acm.org/citation.cfm?id=1162264
// Additionally some of the Viterbi algorithm, is from Wikipedia:
//      https://en.wikipedia.org/wiki/Viterbi_algorithm
//

function pick (xTensor, indexTensor) {
  const slice = tf.gather(xTensor, tf.expandDims(indexTensor, 0));
  return tf.reshape(slice, slice.shape.slice(1));
}

class Gaussian {
  constructor ({ mu, Sigma }) {
    this._states = mu.shape[0];
    this._dimensions = mu.shape[1];

    this._mu = mu;
    this._Sigma = Sigma;

    // TensorFlow.JS can't compute the inverse of a Matrix, so do that
    // externally using ndarray.js
    this._SigmaInv = tf.variable(
      tf.zeros([this._states, this._dimensions, this._dimensions], 'float32'),
      false, null, 'float32'
    );
    // TensorFlow.JS can't compute the determinant of a Matrix, so do that
    // externally using ndarray.js
    this._DetSigma = tf.variable(
      tf.zeros([this._states], 'float32'),
      false, null, 'float32'
    );
    // TensorFlow.JS can't compute the Cholesky factorization of a Matrix,
    // so do that externally using ndarray.js
    this._cholSigma = tf.variable(
      tf.zeros([this._states, this._dimensions, this._dimensions], 'float32'),
      false, null, 'float32'
    );
  }

  async update (finalize = false) {
    const SigmaSafe = tf.tidy(() => tf.maximum(this._Sigma, tf.scalar(1e-16)));
    const matrix = ndarray(await SigmaSafe.data(), this._Sigma.shape);
    this._updateDetSigma(matrix);
    this._updateSigmaInv(matrix);
    if (finalize) {
      this._updateCholSigma(matrix);
    }
  }

  _updateDetSigma (SigmaView) {
    // Uses ndarray-det to compute the determinant of Sigma
    const newDetSigma = new Float32Array(this._states);

    for (let state = 0; state < this._states; state++) {
      const stateSigma = SigmaView.pick(state, null, null);
      const stateDetSigma = ndarrayDet(stateSigma);
      newDetSigma[state] = stateDetSigma;
    }

    this._DetSigma.assign(
      tf.tensor(newDetSigma, [this._states], 'float32')
    );
  }

  _updateSigmaInv (SigmaView) {
    // Uses ndarray-inv to compute the inverse of Sigma
    const newSigmaInv = new Float32Array(
      this._states * this._dimensions * this._dimensions
    );
    const newSigmaInvView = ndarray(
      newSigmaInv,
      [this._states, this._dimensions, this._dimensions]
    );

    for (let state = 0; state < this._states; state++) {
      const stateSigma = SigmaView.pick(state, null, null);
      const stateSigmaInv = ndarrayInv(stateSigma);

      for (let row = 0; row < this._dimensions; row++) {
        for (let col = 0; col < this._dimensions; col++) {
          newSigmaInvView.set(state, row, col, stateSigmaInv.get(row, col));
        }
      }
    }

    this._SigmaInv.assign(tf.tensor(
      newSigmaInv,
      [this._states, this._dimensions, this._dimensions],
      'float32'
    ));
  }

  _updateCholSigma (SigmaView) {
    // Uses ndarray-inv to compute the inverse of Sigma
    const newCholSigma = new Float32Array(
      this._states * this._dimensions * this._dimensions
    );
    const newCholSigmaView = ndarray(
      newCholSigma,
      [this._states, this._dimensions, this._dimensions]
    );

    const placeholderL = ndarray(
      new Float32Array(this._dimensions * this._dimensions),
      [this._dimensions, this._dimensions]
    );

    for (let state = 0; state < this._states; state++) {
      const stateSigma = SigmaView.pick(state, null, null);
      ndarrayCholesky(stateSigma, placeholderL);

      for (let row = 0; row < this._dimensions; row++) {
        for (let col = 0; col < this._dimensions; col++) {
          newCholSigmaView.set(state, row, col, placeholderL.get(row, col));
        }
      }
    }

    this._cholSigma.assign(tf.tensor(
      newCholSigma,
      [this._states, this._dimensions, this._dimensions],
      'float32'
    ));
  }

  pdf (data) {
    // Reshape into [obs, D]
    const dataCollapsed = tf.reshape(data, [-1, this._dimensions]);

    // Implements the PDF of a Multivariate Normal Distribution
    //    see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    const normalizer = tf.mul(
      Math.pow(2 * Math.PI, -(this._dimensions / 2)),
      tf.pow(this._DetSigma, -0.5)
    );

    // Compute the unscaled exponent for each state
    const mu = tf.unstack(this._mu);
    const SigmaInv = tf.unstack(this._SigmaInv);
    const unscaledExponenentList = [];
    for (let state = 0; state < this._states; state++) {
      const x_m_mu = tf.sub(dataCollapsed, tf.expandDims(mu[state], 0));
      const x_m_mu_SigmaInv = tf.matMul(x_m_mu, SigmaInv[state]);
      const x_m_mu_SigmaInv_x_m_mu_T = tf.sum(tf.mul(x_m_mu_SigmaInv, x_m_mu), 1);
      unscaledExponenentList.push(x_m_mu_SigmaInv_x_m_mu_T);
    }
    const unscaledExponent = tf.transpose(tf.stack(unscaledExponenentList));
    const exponent = tf.mul(-0.5, unscaledExponent);
    const pdf = tf.mul(tf.expandDims(normalizer, 0), tf.exp(exponent));

    // Restore original shape
    return tf.reshape(pdf, [...data.shape.slice(0, -1), this._states]);
  }

  sample (states, { seed }) {
    const sample_list = [];
    const states_list = tf.unstack(states);
    const normal = tf.randomNormal(
      [states.shape[0], this._dimensions], 0, 1, 'float32', seed
    );
    const normal_list = tf.unstack(normal);

    for (let n = 0; n < states.shape[0]; n++) {
      const state = states_list[n];
      const mu = pick(this._mu, state);
      const chol = pick(this._cholSigma, state);

      // NOTE: https://en.wikipedia.org/wiki/Multivariate_normal_distribution \
      //       #Drawing_values_from_the_distribution
      const sample = tf.add(
        tf.expandDims(mu, -1),
        tf.matMul(chol, tf.expandDims(normal_list[n], -1))
      );
      sample_list.push(
        tf.reshape(sample, [this._dimensions])
      );
    }
    return tf.stack(sample_list);
  }
}

module.exports = Gaussian;
