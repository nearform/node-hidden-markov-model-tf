
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('expectation step in EM-algorithm', async function (t) {
  const info = require('./expectation.json');
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
  const [alpha, c, beta, gamma, xi] = tf.tidy(() => {
    const emissions = tf.tensor(info.input.emissions);
    const pdf = hmm._gaussian.pdf(emissions);
    const [alpha_hat, c] = hmm._forward(pdf);
    const beta_hat = hmm._backward(pdf, c);
    const [gamma, xi] = hmm._expectation(pdf);
    return [alpha_hat, c, beta_hat, gamma, xi];
  });

  // Check that data approximatly equal
  allclose(t, ndarray(await alpha.data(), alpha.shape), info.output.alpha);
  allclose(t, ndarray(await c.data(), c.shape), info.output.c);
  allclose(t, ndarray(await beta.data(), beta.shape), info.output.beta);

  allclose(t, ndarray(await gamma.data(), gamma.shape), info.output.gamma);
  allclose(t, ndarray(await xi.data(), xi.shape), info.output.xi);
});
