
const isndarray = require('isndarray');
const ndarrayUnpack = require('ndarray-unpack');
const allclose = require('test-allclose');

function allcloseWrapper (t, a, b, { rtol = 1e-05, atol = 1e-08 } = {}) {
  a = isndarray(a) ? ndarrayUnpack(a) : a;
  b = isndarray(b) ? ndarrayUnpack(b) : b;

  allclose(t)(a, b, atol, rtol);
}

module.exports = allcloseWrapper;
