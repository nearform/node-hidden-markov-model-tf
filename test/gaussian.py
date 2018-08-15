
import os.path as path
import json
import tensorflow as tf
import tensorflow_probability as tfp
from .tool_generate_data import GenerateData

thisdir = path.dirname(path.realpath(__file__))

# Make data
generator = GenerateData()
states, emissions = generator.data()

# Compute properbility
tf.enable_eager_execution()
data_tf = tf.constant(emissions)
distributions = tfp.distributions.MultivariateNormalFullCovariance(
    loc=tf.constant(generator.mu),
    covariance_matrix=tf.constant(generator.Sigma)
)
emissions_pdf = distributions.prob(tf.expand_dims(data_tf, -2))


# Save input and output
with open(path.join(thisdir, 'gaussian.json'), 'w') as fp:
    json.dump({
        'config': generator.config,
        'input': {
            'mu': generator.mu.tolist(),
            'Sigma': generator.Sigma.tolist(),
            'emissions': emissions.tolist()
        },
        'output': emissions_pdf.numpy().tolist()
    }, fp)
