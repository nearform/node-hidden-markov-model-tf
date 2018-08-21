
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('log_likelihood by using forward algorithm', async function (t) {
  const info = require('./log_likelihood.json');
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
  const logLikelihood = tf.tidy(() => {
    const data = tf.tensor(info.input.emissions);
    return hmm.logLikelihood(tf.transpose(data, [1, 0, 2]));
  });
  const logLikelihoodView = ndarray(
    await logLikelihood.data(),
    logLikelihood.shape
  );

  // Check that data approximatly equal
  allclose(t, logLikelihoodView, info.output, { rtol: 1e-04, atol: 1e-1 });
});
