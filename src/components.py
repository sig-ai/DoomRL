import numpy as np
from random import sample


class ReplayBuffer(object):
    """
    Class used to sample experiences from the past for training.
    """

    def __init__(self, s_shape, a_shape, capacity, batch_size=4):
        """
        Initializes a replay buffer that can store `capacity`
        experience tuples.

        Each experience is of the form (s1, s2, a, r, t).

        s1 is the original state
        a is the action taking from s1
        r is the resulting reward
        t is whether the state is terminal

        batch_size is the defaut batch size of samples.
        """
        self.capacity = capacity
        self.batch_size = batch_size

        self.s1 = np.zeros([capacity] + (s_shape))
        self.s2 = np.zeros([capacity] + (s_shape))
        self.a = np.zeros([capacity] + (a_shape))
        self.r = np.zeros(capacity)
        self.t = np.zeros(capacity)
        self.idx = 0

    def add_experience(self, s1, s2, a, r, t):
        """
        Adds an experience tuple to the ReplayBuffer.
        """
        idx = self.idx
        self.s1[idx] = s1
        self.s2[idx] = s2
        self.a[idx] = a
        self.r[idx] = r
        self.t[idx] = t
        self.idx = (idx + 1) % self.capacity

    def sample(self, n=None):
        """
        Produces a batch of experiences.

        If `n` is not specified, will use the default batch size.
        """
        n = n or self.batch_size
        idxs = sample(xrange(self.capacity), n)
        batch = self.s1[idxs], self.s2[idxs], self.a[idxs], self.r[idxs]
        return batch
