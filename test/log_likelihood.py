
import os.path as path
import json
import numpy as np
import tensorflow as tf
from .tool_generate_data import GenerateData
from .hmm import HMM

thisdir = path.dirname(path.realpath(__file__))

generator = GenerateData(num_time=7)

# Make data
states, emissions = generator.data()

# Check reference
hmm = HMM(generator.num_states, generator.num_dims,
          obs=generator.num_obs, time=generator.num_time)
hmm._p0[:, :] = generator.pi[np.newaxis, :]
hmm._tp[:, :] = generator.A
hmm._mu[:, :] = generator.mu
hmm._sigma[:, :, :] = generator.Sigma
log_likelihood = hmm.posterior(np.transpose(emissions, [1, 0, 2]))

# Save input and output
with open(path.join(thisdir, 'log_likelihood.json'), 'w') as fp:
    json.dump({
        'config': {
            **generator.config
        },
        'input': {
            'emissions': emissions.tolist(),
            'pi': generator.pi.tolist(),
            'A': generator.A.tolist(),
            'mu': generator.mu.tolist(),
            'Sigma': generator.Sigma.tolist()
        },
        'output': log_likelihood.tolist()
    }, fp)
