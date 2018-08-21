
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('maximization step in EM-algorithm', async function (t) {
  const info = require('./maximization.json');
  const hmm = new HMM({
    states: info.config.states,
    dimensions: info.config.dimensions
  });

  await hmm.setParameters({
    pi: tf.tensor(info.input.pi),
    A: tf.tensor(info.input.A),
    mu: tf.tensor(info.input.mu),
    Sigma: tf.tensor(info.input.Sigma)
  });

  // Precompute pdf for the data
  const [pi, A, mu, Sigma] = tf.tidy(() => {
    const data = tf.tensor(info.input.emissions);
    const pdf = hmm._gaussian.pdf(data);
    const [gamma, xi] = hmm._expectation(pdf);
    const [pi, A, mu, Sigma] = hmm._maximization(data, gamma, xi);
    return [pi, A, mu, Sigma];
  });

  // Check that data approximatly equal
  allclose(t, ndarray(await pi.data(), pi.shape), info.output.pi);
  allclose(t, ndarray(await A.data(), A.shape), info.output.A);
  allclose(t, ndarray(await mu.data(), mu.shape), info.output.mu);
  allclose(t, ndarray(await Sigma.data(), Sigma.shape), info.output.Sigma);
});
