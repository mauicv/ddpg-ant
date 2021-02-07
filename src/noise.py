
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d


class NormalNoise:
    def __init__(
            self,
            dim,
            sigma=0.2,
            dt=1e-2):
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
        self.g = tf.random.Generator.from_seed(1)
        self.reset()

    def __call__(self):
        return self.sigma * np.sqrt(self.dt) * \
            self.g.normal([self.dim], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)

    def reset(self):
        return


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from https://keras.io/examples/rl/ddpg_pendulum/
    Formula from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """
    def __init__(
            self,
            dim=1,
            sigma=0.15,
            theta=0.2,
            dt=1e-2,
            x_initial=None):
        self.theta = theta
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.g = tf.random.Generator.from_seed(1)
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (- self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt)
            * self.g.normal([self.dim], mean=0.0, stddev=1.0,
                            dtype=tf.dtypes.float32)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.dim)
