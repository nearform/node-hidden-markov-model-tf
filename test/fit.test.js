
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const ndarrayUnpack = require('ndarray-unpack');
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('fit uses EM-algorithm correctly', async function (t) {
  const info = require('./fit.json');
  const hmm = new HMM({
    states: info.config.states,
    dimensions: info.config.dimensions
  });

  const emissions = tf.transpose(tf.tensor(info.input), [1, 0, 2]);
  const results = await hmm.fit(emissions, { tolerance: 0.0001 });
  const { pi, A, mu, Sigma } = hmm.getParameters();

  t.ok(results.tolerance < 0.0001);
  t.ok(results.iterations < 10);
  t.ok(results.converged);

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

  // Check that data approximatly equal
  allclose(t, piSorted, info.output.hmmlearn.pi);
  allclose(t, ASorted, info.output.hmmlearn.A);
  allclose(t, muSorted, info.output.hmmlearn.mu);
  allclose(t, SigmaSorted, info.output.hmmlearn.Sigma,
    { rtol: 1e-5, atol: 1e-4 });

  allclose(t, piSorted, info.output.tensorflow.pi);
  allclose(t, ASorted, info.output.tensorflow.A);
  allclose(t, muSorted, info.output.tensorflow.mu);
  allclose(t, SigmaSorted, info.output.tensorflow.Sigma,
    { rtol: 1e-5, atol: 1e-4 });
});
