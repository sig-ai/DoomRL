from __future__ import division
import numpy as np
from random import sample


class ReplayBuffer(object):
    """
    Class used to sample experiences from the past for training.
    """

    def __init__(self, input_learner, env, capacity, a_shape=[], batch_size=32):
        """
        Initializes a replay buffer that can store `capacity`
        experience tuples.

        Each experience is of the form (ob, next_ob, a, r, t).

        `ob` is the original observation
        `a` is the action taking from ob
        `next_ob` is the next observation
        `r` is the resulting reward
        `t` is whether the state is terminal

        `batch_size` is the size of samples.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.input_learner = input_learner

        s_shape = env.observation_space.shape
        self.ob = np.zeros([capacity] + list(s_shape))
        self.next_ob = np.zeros([capacity] + list(s_shape))
        self.a = np.zeros([capacity] + list(a_shape))
        self.r = np.zeros(capacity)

        self.size = 0
        self.idx = 0

    def add_experience(self, ob, next_ob, a, r):
        """
        Adds an experience tuple to the ReplayBuffer.
        """
        idx = self.idx
        self.ob[idx] = ob
        self.next_ob[idx] = next_ob
        self.a[idx] = a
        self.r[idx] = r
        self.idx = (idx + 1) % self.capacity
        self.size = max(self.idx, self.size)

    def filled(self):
        """
        Returns whether the ReplayBuffer has at least `batch_size` 
        experiences.
        """
        return self.size > self.batch_size

    def sample(self):
        """
        Produces a batch of experiences.
        """
        assert(self.filled())
        n = self.batch_size
        idxs = sample(xrange(self.size), n)
        batch = self.ob[idxs], self.next_ob[idxs], self.a[idxs], self.r[idxs]
        return batch

    def get_learner(self):
        def learner(ob, next_ob, a, r):
            self.add_experience(ob, next_ob, a, r)
            if self.filled():
                a,b,c,d = self.sample()
                self.input_learner(a,b,c,d)
        return learner
