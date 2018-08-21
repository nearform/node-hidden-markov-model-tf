
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const ndarrayUnpack = require("ndarray-unpack");
const tf = require('./tensorflow.js');

const Initialize = require('../lib/initialize.js');

test('KMeans Initialization for mu and Sigma', async function (t) {
  const info = require('./initialize.json');
  const initialize = new Initialize({
    states: info.config.states,
    dimensions: info.config.dimensions
  });

  const emissions = tf.tensor(info.input);
  const [mu, Sigma] = await initialize.compute(emissions, {
    maxIterations: info.config.maxIterations,
    tolerance: info.config.tolerance,
    seed: 2
  });

  const muView = ndarray(await mu.data(), mu.shape);
  const SigmaView = ndarray(await Sigma.data(), Sigma.shape);

  // Compute sort indices
  const sortInput = [];
  for (let s = 0; s < info.config.states; s++) {
    sortInput.push({
      index: s,
      value: muView.get(s, 0)
    });
  }
  const sortIndices = sortInput.sort(function (a, b) {
    return a.value - b.value;
  }).map((o) => o.index);

  // Reorder muView and SigmaView
  const muSorted = [];
  const SigmaSorted = [];
  for (let s = 0; s < info.config.states; s++) {
    const oldIndex = sortIndices[s];
    muSorted.push(ndarrayUnpack(muView.pick(oldIndex, null)));
    SigmaSorted.push(ndarrayUnpack(SigmaView.pick(oldIndex, null, null)));
  }

  // Check that data approximatly equal
  allclose(t, muSorted, info.output.mu, { rtol: 1e-5, atol: 0.5 });
  allclose(t, SigmaSorted, info.output.Sigma, { rtol: 1e-5, atol: 0.5 });
});
