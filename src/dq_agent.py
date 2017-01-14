from __future__ import division

import gym
import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import losses, slim
from keras.layers import Input, Conv2D, Lambda
from keras.utils.np_utils import to_categorical
import numpy as np
from random import sample
from keras.layers import Merge
from keras.models import Sequential
import keras.backend as K




class ReplayBuffer(object):
    """
    Class used to sample experiences from the past for training.
    """

    def __init__(self, ob_shape, num_actions, warmup_steps=50000,
                 capacity=100000, batch_size=128):
        """
        Initializes a replay buffer that can store `capacity`
        experience tuples.

        Returned experience is of the form (ob, next_ob, a, r, t).

        `ob` is a given observation, and `next_ob` is the subsequent obvsersation
        `a` is the action taken from `ob`
        `r` is the resulting reward
        `t` is whether the state is terminal

        `batch_size` is the size of samples.
        """
        self.capacity = capacity
        self.batch_size = batch_size

        self.ob = np.zeros([capacity] + list(ob_shape))
        self.a = np.zeros(capacity)
        self.r = np.zeros(capacity)
        self.t = np.zeros(capacity)

        self.size = 0
        assert(warmup_steps <= capacity)
        self.warmup_steps = warmup_steps
        self.idx = 0

    def add_experience(self, ob, a, r, t):
        """
        Adds an experience tuple to the ReplayBuffer.
        """
        idx = self.idx
        self.ob[idx] = ob
        self.a[idx] = a
        self.r[idx] = r
        self.t[idx] = t

        self.idx = (idx + 1) % self.capacity
        self.size = max(self.idx, self.size)

    def ready(self):
        """
        Returns whether the ReplayBuffer has at least `batch_size` 
        experiences.
        """
        return self.size > max(self.batch_size, self.warmup_steps)

    def sample(self):
        """
        Produces a batch of experiences.
        Returns `None` if  there are less than `batch_size` experiences.
        """
        if not (self.ready()):
            return None
        idxs = np.array(sample(xrange(self.size), self.batch_size))
        batch = self.ob[idxs], self.ob[
            idxs + 1], self.a[idxs], self.r[idxs], self.t[idxs + 1]
        return batch


from skimage.color import rgb2gray
from skimage.transform import resize

from keras.models import Model
from keras.optimizers import RMSprop

def wrap_model(net, num_a):
    mask_input = Input([num_a])
    mask = Model(mask_input, mask_input)
    a_vals = Merge([net, mask],  name='a_vals', output_shape=[1],
                   mode=lambda x: K.sum(x[0] * x[1], 1))
    wrapped = Model(input=[net.input, mask.input], output=a_vals.output)
    return wrapped


class DQAgent(object):

    def __init__(self, model, num_actions, ob_shape, discount_factor=.99,
                 sync_steps=10000, warmup_steps=50000, steps_per_update=4):
        self.num_actions = num_actions
        self.obs_shape = ob_shape
        self.online = model
        self.online_a = wrap_model(self.online, self.num_actions)
        self.target = Sequential.from_config(model.get_config())
        self.target_a = wrap_model(self.target, self.num_actions)
        self._sync()

        self.discount_factor = discount_factor
        self.warmed_up = False
        op = RMSprop(lr=0.00025)

        self.online_a.compile(op, 'mse')
        self.mem = ReplayBuffer(ob_shape, num_actions)

        self.steps = 0
        self.sync_steps = int(sync_steps)
        self.warmup_steps = warmup_steps
        self.steps_per_update = 4

    def _sync(self):
        self.target.set_weights(self.online.get_weights())

    def select_action(self, ob, batch=False):
        if not batch:
            ob = [ob]
        q_vals = self.online.predict(np.array(ob))
        action = np.argmax(q_vals)
        return action

    def learn(self, ob, a, r, t, updates=4):
        self.mem.add_experience(ob, a, r, t)
        self.steps += 1
        if self.steps % self.steps_per_update == 0:
            self._update_model()
        if self.steps % self.sync_steps == 0:
            print('Syncing target/online')
            self._sync()

    def _update_model(self):
        if self.mem.ready():
            self.warmed_up = True
            o1, o2, a1, r, t = self.mem.sample()
            r = np.clip(r, -1, 1)
            mask = to_categorical(a1, self.num_actions)
            discounting = self.discount_factor * (1 - t)
            a2 = self.select_action(o2, batch=True)
            target_vals = r + self.target_a.predict([o2,a2]) * discounting
            self.online_a.train_on_batch([o1, mask], target_vals)
