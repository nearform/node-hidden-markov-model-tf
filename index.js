
const semver = require('semver');
const tfrange = require('./package.json')['devDependencies']['@tensorflow/tfjs-core'];

let tf;
try {
  tf = require('@tensorflow/tfjs-core');
} catch (_) {
  exit('@tensorflow/tfjs-core could not be found');
}

if (!semver.satisfies(tf.version_core, tfrange)) {
  exit(`version ${tf.version_core} of @tensorflow/tfjs-core did not match ${tfrange}`);
}

function exit(condition) {
  console.error(condition);
  console.error(`Please run: npm install --save @tensorflow/tfjs-core@"${tfrange}"`);
  process.exit(1);
}

module.exports = require('./lib/hmm.js');
