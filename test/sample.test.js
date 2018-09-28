
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const ndarrayUnpack = require('ndarray-unpack');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('sampling from HMM distribution', async function (t) {
  const info = require('./sample.json');
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

  // Sample data given the distribution
  const { emissions } = hmm.sample({
    observations: info.config.observations,
    time: info.config.time,
    seed: 1
  });

  // Fit model for the data
  const results = await hmm.fit(emissions, { seed: 1 });
  t.ok(results.converged);

  // Get parameters for the fitted model
  const { pi, A, mu, Sigma } = hmm.getParameters();

  // Fetch data and make a view
  const piView = ndarray(await pi.data(), pi.shape);
  const AView = ndarray(await A.data(), A.shape);
  const muView = ndarray(await mu.data(), mu.shape);
  const SigmaView = ndarray(await Sigma.data(), Sigma.shape);

  // Compute sort indices, by sorting \pi
  const sortInput = [];
  for (let s = 0; s < info.config.states; s++) {
    sortInput.push({
      index: s,
      value: piView.get(s)
    });
  }
  const sortIndices = sortInput.sort(function (a, b) {
    return a.value - b.value;
  }).map((o) => o.index);

  // Reorder parameters
  const piSorted = ndarray(new Float32Array(pi.shape[0]), pi.shape);
  const ASorted = ndarray(new Float32Array(A.shape[0] * A.shape[1]), A.shape);
  const muSorted = [];
  const SigmaSorted = [];
  for (let s = 0; s < info.config.states; s++) {
    const oldIndex = sortIndices[s];
    piSorted.set(s, piView.get(oldIndex));
    for (let s2 = 0; s2 < info.config.states; s2++) {
      ASorted.set(s, s2, AView.get(oldIndex, sortIndices[s2]));
    }
    muSorted.push(ndarrayUnpack(muView.pick(oldIndex, null)));
    SigmaSorted.push(ndarrayUnpack(SigmaView.pick(oldIndex, null, null)));
  }

  // Check that the fitted parameters from the sampled data matches
  // the input parameters.
  allclose(t, piSorted, info.input.pi, { rtol: 1e-5, atol: 1e-1 });
  allclose(t, ASorted, info.input.A, { rtol: 1e-5, atol: 1e-1 });
  allclose(t, muSorted, info.input.mu, { rtol: 1e-5, atol: 1e-1 });
  allclose(t, SigmaSorted, info.input.Sigma, { rtol: 1e-5, atol: 1e-1 });
});
