
const test = require('tap').test;
const allclose = require('./allclose.js');
const ndarray = require('ndarray');
const tf = require('./tensorflow.js');

const Gaussian = require('../lib/gaussian.js');

test('test Gaussian PDF calculations', async function (t) {
  const info = require('./gaussian.json');
  const gaussian = tf.tidy(() => new Gaussian({
    mu: tf.tensor(info.input.mu),
    Sigma: tf.tensor(info.input.Sigma)
  }));

  await gaussian.update();

  const pdf = tf.tidy(() => {
    const emissions = tf.tensor(info.input.emissions);
    return gaussian.pdf(emissions);
  });
  const pdfView = ndarray(await pdf.data(), pdf.shape);

  allclose(t, pdfView, info.output, { rtol: 1e-01, atol: 1e-09 });
});
