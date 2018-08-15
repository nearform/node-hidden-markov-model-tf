
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return e_x / np.sum(e_x, axis=1)[:, np.newaxis]


class GenerateData:
    def __init__(self,
                 num_obs=5, num_time=27, num_states=3, num_dims=2, seed=1):
        self.num_obs = num_obs
        self.num_time = num_time
        self.num_states = num_states
        self.num_dims = num_dims
        self._rng = np.random.RandomState(seed)

        mu = []
        Sigma = []
        for s in range(num_states):
            cov_undefinte = self._rng.uniform(-1, 1, size=(num_dims, num_dims))

            mu.append(self._rng.uniform(-10, 10, size=num_dims))
            Sigma.append(0.5 * (cov_undefinte @ cov_undefinte.T))

        self.mu = np.stack(mu)
        self.Sigma = np.stack(Sigma)
        self.pi = softmax(
            self._rng.uniform(-1, 1, size=(1, self.num_states))
        ).ravel()
        self.A = softmax(
            self._rng.uniform(-1, 1, size=(self.num_states, self.num_states))
        )

    @property
    def config(self):
        return {
            'observations': self.num_obs,
            'time': self.num_time,
            'states': self.num_states,
            'dimensions': self.num_dims
        }

    def data(self):
        emissions = np.zeros(
            (self.num_time, self.num_obs, self.num_dims),
            dtype=np.float64
        )
        states = np.zeros(
            (self.num_time, self.num_obs),
            dtype=np.int32
        )

        for n in range(self.num_obs):
            state_0 = self._rng.choice(self.num_states,
                                       p=self.pi)
            emissions[0, n, :] = self._rng.multivariate_normal(
                self.mu[state_0, :], self.Sigma[state_0, :, :]
            )

            states[0, n] = state_0
            for t in range(1, self.num_time):
                state_tm1 = states[t - 1, n]
                state_t = self._rng.choice(self.num_states,
                                           p=self.A[state_tm1, :])
                emissions[t, n, :] = self._rng.multivariate_normal(
                    self.mu[state_t, :], self.Sigma[state_t, :, :]
                )
                states[t, n] = state_t

        return states, emissions
