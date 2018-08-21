
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('inference using viterbi', async function (t) {
  const info = require('./inference.json');
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
  const states = tf.tidy(() => {
    const data = tf.tensor(info.input.emissions);
    return hmm.inference(tf.transpose(data, [1, 0, 2]));
  });
  const statesView = ndarray(await states.data(), states.shape);

  // Check that data approximatly equal
  allclose(t, statesView, info.output.hmmlearn);
  allclose(t, statesView, info.output.tensorflow);
});
