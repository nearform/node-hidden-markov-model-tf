# hidden-markov-model-tf

[![Greenkeeper badge](https://badges.greenkeeper.io/nearform/node-hidden-markov-model-tf.svg)](https://greenkeeper.io/)

**A trainable Hidden Markov Model with Gaussian emissions using TensorFlow.js**

## Install

```
$ npm install hidden-markov-model-tf
```

Require: Node v12+

## Usage

```js
const assert = require('assert'):
require('@tensorflow/tfjs-node'); // Optional, enable native TensorFlow backend
const tf = require('@tensorflow/tfjs');
const HMM = require('hidden-markov-model-tf');

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
    [-1.5,  3.7],
    [-1.7,  1.2]
  ]),
  Sigma: tf.tensor([
    [[ 0.12, -0.01],
     [-0.01,  0.50]],
    [[ 0.21,  0.05],
     [ 0.05,  0.03]],
    [[ 0.37,  0.35],
     [ 0.35,  0.44]]
  ])
});

// Sample data
const sample = hmm.sample({observations, time});
assert.deepEqual(sample.states.shape, [observations, time]);
assert.deepEqual(sample.emissions.shape, [observations, time, dimensions]);

// Your data must be a tf.tensor with shape [observations, time, dimensions]
const data = sample.emissions;

// Fit model with data
const results = await hmm.fit(data);
assert(results.converged);

// Predict hidden state indices
const inference = hmm.inference(data);
assert.deepEqual(inference.shape, [observations, time]);
states.print();

// Compute log-likelihood
const logLikelihood = hmm.logLikelihood(data);
assert.deepEqual(logLikelihood.shape, [observations]);
logLikelihood.print();

// Get parameters
const {pi, A, mu, Sigma} = hmm.getParameters();
pi.print();
A.print();
mu.print();
Sigma.print();
```

## Documentation

`hidden-markov-model-tf` is TensorFlow.js based, therefore your input must
be povided as a `tf.tensor`. Likewise most outputs are also provided as a
`tf.tensor`. You can always get a `TypedArray` with `await tensor.data()`.

### hmm = new HMM({states, dimensions})

The constructor takes two integer arguments. The number of hidden `states` and
the number of `dimensions` in the Gaussian emissions.

### result = await hmm.fit(tensor, {maxIterations = 100, tolerance = 0.001, seed})

The `fit` method, takes an required `tf.tensor` object. That must have the
shape `[observations, time, dimensions]`. If you only have one observation
it should have the shape `[1, time, dimensions]`.

The `fit` method, returns a `Promise` for the `results`. The `results` is
an object with the following properties:

```js
const {
   // the number of iterations used, will at most be `maxIterations`
  iterations,

   // if the training coverged, given the `tolerance`,
   // before `maxIterations` was reached
  converged,

  // The achived tolerance, after the number of iterations. This can be
  // useful if the optimizer did not converge, but you want to know how
  // good the fit is.
  tolerance
} = await hmm.fit(tensor);
```

The `fit` method uses a KMeans initialization. This initialization algorithm is
random but can be seeded with the optional `seed` parameter.

After initialization, the model is optimized using an EM-algorithm called
the [Baum–Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm).

### states = hmm.inference(tensor)

The `inference` method, takes an required `tf.tensor` object. That must have
the shape `[observations, time, dimensions]`.

It uses the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
for infering the hidden state. Which is returned as `tf.tensor` with the
shape `[observations, time]`.

```js
const states = hmm.inference(tensor);
states.print();
console.log(await states.data());
```

### logLikelihood = hmm.logLikelihood(tensor)

The `inference` method, takes an required `tf.tensor` object. That must have
the shape `[observations, time, dimensions]`.

It uses the forward procedure of the
[Baum–Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
to compute the logLikelihood for each observation. This is returned as a
`tf.tensor` with the shape `[observations]`.

### {states, emissions} = hmm.sample({ observations, time, seed })

The `sample` method, samples data from the Hidden Markov Model distribution
and returns both the sampled states and Gaussian emissions, as two `tf.tensor`
objects.

the `states` tensor has the shape `[observations, time]`. While the `emissions`
tensor has the `shape` [observations, time, dimensions].

The sampling can be seed with the optional `seed` parameter.

### {pi, A, mu, Sigma} = hmm.getParameters()

Return the underlying parameters:

* `pi`: the hidden state prior distribution. `shape = [states]`
* `A`: the hidden state transfer distribution. `shape = [states, states]`
* `mu`: the mean of the Gaussian emission distribution. `shape = [states, dimensions]`
* `Sigma`: the covariance matrix of the Gaussian emission distribution. `shape = [states, dimensions,  dimensions]`

### await hmm.setParameters({pi, A, mu, Sigma})

Set the underlying parameters of the Hidden Markov Model. Note that some
internal properties related to the Gaussian distribution will be precomputed.
Therefore this returns a `Promise`. Be sure to wait for the promise to
resolve before calling any other method.
