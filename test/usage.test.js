
const test = require('tap').test;
const tf = require('./tensorflow.js');

const HMM = require('../lib/hmm.js');

test('usage documentation', async function (t) {
  const [observations, time, states, dimensions] = [5, 7, 3, 2];

  // Configure model
  const hmm = new HMM({
    states: states,
    dimensions: dimensions
  });

  // Set parameters
  await hmm.setParameters({
    pi: tf.tensor([0.15, 0.20, 0.65]),
    A: tf.tensor([
      [0.55, 0.15, 0.30],
      [0.45, 0.45, 0.10],
      [0.15, 0.20, 0.65]
    ]),
    mu: tf.tensor([
      [-7.0, -8.0],
      [-1.5, 3.7],
      [-1.7, 1.2]
    ]),
    Sigma: tf.tensor([
      [[0.12, -0.01],
        [-0.01, 0.50]],
      [[0.21, 0.05],
        [0.05, 0.03]],
      [[0.37, 0.35],
        [0.35, 0.44]]
    ])
  });

  // Sample data
  const sample = hmm.sample({ observations, time });
  t.deepEqual(sample.states.shape, [observations, time]);
  t.deepEqual(sample.emissions.shape, [observations, time, dimensions]);

  // Your data must be a tf.tensor with shape [observations, time, dimensions]
  const data = sample.emissions;

  // Fit model with data
  const results = await hmm.fit(data);
  t.ok(results.converged);

  // Predict hidden state indices
  const inference = hmm.inference(data);
  t.deepEqual(inference.shape, [observations, time]);

  // Compute log-likelihood
  const logLikelihood = hmm.logLikelihood(data);
  t.deepEqual(logLikelihood.shape, [observations]);

  // Get parameters
  const { pi, A, mu, Sigma } = hmm.getParameters();
  t.deepEqual(pi.shape, [states]);
  t.deepEqual(A.shape, [states, states]);
  t.deepEqual(mu.shape, [states, dimensions]);
  t.deepEqual(Sigma.shape, [states, dimensions, dimensions]);
});

test('usage mistakes in states', async function (t) {
  // Uneven numbers should not be accepted as states.
  t.throws(
    () => new HMM({
      states: 1.5,
      dimensions: 2
    }),
    TypeError
  );
  // Other types than number should not be accepted as states.
  t.throws(
    () => new HMM({
      states: '1',
      dimensions: 2
    }),
    TypeError
  );
  // Negative or zero states should not be accepted.
  t.throws(
    () => new HMM({
      states: 0,
      dimensions: 2
    }),
    TypeError
  );
});

test('usage mistakes in dimensions', async function (t) {
  // Uneven numbers should not be accepted as states.
  t.throws(
    () => new HMM({
      states: 1,
      dimensions: 1.5
    }),
    TypeError
  );
  // Other types than number should not be accepted as states.
  t.throws(
    () => new HMM({
      states: 1,
      dimensions: '1'
    }),
    TypeError
  );
  // Negative states should not be accepted.
  t.throws(
    () => new HMM({
      states: 1,
      dimensions: -1
    }),
    TypeError
  );
});
