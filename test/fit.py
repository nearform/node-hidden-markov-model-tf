
import json
import os.path as path
import numpy as np
from hmmlearn import hmm
from .hmm import HMM

from .tool_generate_data import GenerateData

thisdir = path.dirname(path.realpath(__file__))

# Make data
generator = GenerateData(num_obs=20)
states, emissions = generator.data()
lengths = np.full([generator.num_obs], generator.num_time)

# Fit hmmlearn model
emissions_collaped = np.transpose(emissions, [1, 0, 2])  # (N, T, D)
emissions_collaped = emissions_collaped.reshape(
    generator.num_obs * generator.num_time, generator.num_dims
)
hmmlearn_model = hmm.GaussianHMM(
    n_components=generator.num_states,
    covariance_type="full"
)
hmmlearn_model.fit(emissions_collaped, lengths)

hmmlearn_reorder = np.argsort(hmmlearn_model.startprob_)
hmmlearn_pi = hmmlearn_model.startprob_[hmmlearn_reorder]
hmmlearn_A = hmmlearn_model.transmat_[hmmlearn_reorder, :][:, hmmlearn_reorder]
hmmlearn_mu = hmmlearn_model.means_[hmmlearn_reorder, :]
hmmlearn_Sigma = hmmlearn_model.covars_[hmmlearn_reorder, :, :]

# Fit TensorFlow model
tf_model = HMM(generator.num_states, generator.num_dims,
               obs=generator.num_obs, time=generator.num_time)
tf_model.fit(np.transpose(emissions, [1, 0, 2]))

tf_reorder = np.argsort(tf_model._p0.ravel())
tf_pi = tf_model._p0.ravel()[tf_reorder]
tf_A = tf_model._tp[tf_reorder, :][:, tf_reorder]
tf_mu = tf_model._mu[tf_reorder, :]
tf_Sigma = tf_model._sigma[tf_reorder, :, :]

# Save input and output
with open(path.join(thisdir, 'fit.json'), 'w') as fp:
    json.dump({
        'config': generator.config,
        'input': emissions.tolist(),
        'output': {
            'hmmlearn': {
                'pi': hmmlearn_pi.tolist(),
                'A': hmmlearn_A.tolist(),
                'mu': hmmlearn_mu.tolist(),
                'Sigma': hmmlearn_Sigma.tolist()
            },
            'tensorflow': {
                'pi': tf_pi.tolist(),
                'A': tf_A.tolist(),
                'mu': tf_mu.tolist(),
                'Sigma': tf_Sigma.tolist()
            }
        }
    }, fp)
