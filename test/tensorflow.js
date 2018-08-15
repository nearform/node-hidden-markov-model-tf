
process.env.TF_CPP_MIN_LOG_LEVEL = '1'; // Supress info logging
require('@tensorflow/tfjs-node'); // Enable Native TensorFlow
const tf = require('@tensorflow/tfjs-core');
// tf.setBackend('tensorflow', true); // Enable safe-mode

module.exports = tf;
