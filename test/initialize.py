
import os.path as path
import json
import numpy as np
from sklearn.cluster import KMeans
from .tool_generate_data import GenerateData

thisdir = path.dirname(path.realpath(__file__))

max_iter = 100
tolerance = 0.0001
generator = GenerateData(seed=2)

# Make data
states, emissions = generator.data()

# Fit data
emissions_timeless = emissions.reshape(
    emissions.shape[0] * emissions.shape[1], emissions.shape[2]
)
kmeans = KMeans(n_clusters=generator.num_states,
                max_iter=max_iter, tol=tolerance)
states = kmeans.fit_predict(emissions_timeless)

# get \mu and \Sigma
mu = kmeans.cluster_centers_
Sigma = np.stack([
    np.cov(emissions_timeless[states == s], rowvar=False)
    for s in range(generator.num_states)
])

# Sort \mu and \Sigma by the first \mu coordiate
reorder_indices = np.argsort(mu[:, 0])
mu = mu[reorder_indices, :]
Sigma = Sigma[reorder_indices, :, :]

# Save input and output
with open(path.join(thisdir, 'initialize.json'), 'w') as fp:
    json.dump({
        'config': {
            **generator.config,
            'maxIterations': max_iter,
            'tolerance': tolerance
        },
        'input': emissions.tolist(),
        'output': {
            'mu': generator.mu.tolist(),
            'Sigma':  generator.Sigma.tolist()
        }
    }, fp)
