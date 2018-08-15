
import os.path as path
import json
import numpy as np
from .tool_generate_data import GenerateData

thisdir = path.dirname(path.realpath(__file__))

max_iter = 100
tolerance = 0.0001
generator = GenerateData(num_obs=181, num_time=21, seed=2)

reorder = np.argsort(generator.pi.ravel())

# Save input and output
with open(path.join(thisdir, 'sample.json'), 'w') as fp:
    json.dump({
        'config': generator.config,
        'input': {
            'pi': generator.pi[reorder].tolist(),
            'A': generator.A[reorder, :][:, reorder].tolist(),
            'mu': generator.mu[reorder, :].tolist(),
            'Sigma': generator.Sigma[reorder, :, :].tolist()
        }
    }, fp)
