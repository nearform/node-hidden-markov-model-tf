
import os.path as path
import json
import numpy as np
import tensorflow as tf
from .tool_generate_data import GenerateData
from hmmlearn import hmm
from .hmm import HMM

thisdir = path.dirname(path.realpath(__file__))

max_iter = 100
tolerance = 0.0001
generator = GenerateData(num_time=7, seed=2)

# Make data
states, emissions = generator.data()
lengths = np.full([generator.num_obs], generator.num_time)

# Check hmmlearn reference
emissions_collaped = np.transpose(emissions, [1, 0, 2])  # (N, T, D)
emissions_collaped = emissions_collaped.reshape(
    generator.num_obs * generator.num_time, generator.num_dims
)

hmmlearn_model = hmm.GaussianHMM(
    n_components=generator.num_states,
    covariance_type="full"
)
hmmlearn_model.n_features = generator.num_dims
hmmlearn_model.startprob_ = generator.pi
hmmlearn_model.transmat_ = generator.A
hmmlearn_model.means_ = generator.mu
hmmlearn_model.covars_ = generator.Sigma

hmmlearn_ref = hmmlearn_model.predict(emissions_collaped, lengths)
hmmlearn_ref = hmmlearn_ref.reshape(generator.num_obs, generator.num_time)

# Check TF reference
tf_model = HMM(generator.num_states, generator.num_dims,
               obs=generator.num_obs, time=generator.num_time)
tf_model._p0[:, :] = generator.pi[np.newaxis, :]
tf_model._tp[:, :] = generator.A
tf_model._mu[:, :] = generator.mu
tf_model._sigma[:, :, :] = generator.Sigma

tensorflow_ref = tf_model.run_viterbi(np.transpose(emissions, [1, 0, 2]))

# Save input and output
with open(path.join(thisdir, 'inference.json'), 'w') as fp:
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
            'hmmlearn': hmmlearn_ref.tolist(),
            'tensorflow': tensorflow_ref.tolist()
        }
    }, fp)
