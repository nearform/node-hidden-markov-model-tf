
import os.path as path
import json
import numpy as np
import tensorflow as tf
from .tool_generate_data import GenerateData
from .hmm import HMM

thisdir = path.dirname(path.realpath(__file__))

max_iter = 100
tolerance = 0.0001
generator = GenerateData(num_time=7, seed=2)

# Make data
states, emissions = generator.data()

# Check reference
hmm = HMM(generator.num_states, generator.num_dims,
          obs=generator.num_obs, time=generator.num_time)
with tf.Session(graph=hmm._graph) as sess:
    sess.run(tf.global_variables_initializer())
    pi, A, mu, Sigma = sess.run(
        [
            hmm._p0_tf_new, hmm._tp_tf_new,
            hmm._mu_tf_new, hmm._sigma_tf_new
        ],
        {
            hmm._dataset_tf: np.transpose(emissions, [1, 0, 2]),
            hmm._p0_tf: generator.pi[np.newaxis, :],
            hmm._tp_tf: generator.A,
            hmm._mu_tf: generator.mu,
            hmm._sigma_tf: generator.Sigma
        }
    )

# Save input and output
with open(path.join(thisdir, 'maximization.json'), 'w') as fp:
    json.dump({
        'config': {
            **generator.config,
            'maxIterations': max_iter,
            'tolerance': tolerance
        },
        'input': {
            'emissions': emissions.tolist(),
            'pi': generator.pi.tolist(),
            'A': generator.A.tolist(),
            'mu': generator.mu.tolist(),
            'Sigma': generator.Sigma.tolist()
        },
        'output': {
            'pi': pi.ravel().tolist(),
            'A': A.tolist(),
            'mu': mu.tolist(),
            'Sigma': Sigma.tolist()
        }
    }, fp)
