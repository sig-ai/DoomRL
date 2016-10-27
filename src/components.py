from __future__ import division
import numpy as np
from random import random, sample

import gym
import gym_pull
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete


class ReplayBuffer(object):
    """
    Class used to sample experiences from the past for training.
    """

    def __init__(self, s_shape, a_shape, capacity, batch_size=32):
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

        self.ob = np.zeros([capacity] + list(s_shape))
        self.next_ob = np.zeros([capacity] + list(s_shape))
        if a_shape is None:
            self.a = np.zeros([capacity])
        else:
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


def basic_env(moves='constant-7', resolution='160x120'):
    """
    Sets up an environment with a discretized action space
    and lower resolution.
    """
    env = gym.make('ppaquette/DoomDeathmatch-v0')
    env = SetResolution(resolution)(env)
    env = ToDiscrete(moves)(env)
    return env


def decay_fn(total_iterations, output_range):
    """
    Returns a function that decays its outputs with respect
    to the iterations. Useful for decreasing exploration and
    learning rates over time.
    """
    first_output, last_output = output_range
    step_update = (last_output - first_output) / total_iterations

    def decay(x):
        """ Output decays linearly with iterations. """
        return first_output + x * step_update
    return decay


def eps_greedy(eps, ob, agent, env):
    """
    With probability `eps`, it will sample a random move.

    Otherwise, returns the agents usual action.
    """
    if random() < eps:
        return env.action_space.sample()
    return agent.act(ob)


def learn_doom(agent, env, episodes=10000, render=False, frame_skip=1,
               replay_buffer_size=500, batch_size=32,
               logdir='/tmp/doom-agent-results',
               learning_rate_range=(.001, .0001), epsilon_range=(.8, .01)):
    """
    Trains the agent in specified env.

    episodes: the number of episodes to run.
    """

    env.monitor.start(logdir, force=True, seed=0)
    print("Outputting results to {0}".format(logdir))

    rb = ReplayBuffer(env.observation_space.shape, None, replay_buffer_size,
                      batch_size)
    get_epsilon = decay_fn(episodes, epsilon_range)
    get_learning_rate = decay_fn(episodes, learning_rate_range)
    show = lambda: env.render() if render else None
    
    for episode in xrange(episodes):
        print("Episode {0}\n".format(episode))
        epsilon = get_epsilon(episode)
        learning_rate = get_learning_rate(episode)

        done = False     # Whether the current episode concluded
        prev_ob = None   # The previous observation
        ob = env.reset()  # The current observation
        total_reward = 0
        while not done:
            action = eps_greedy(epsilon, ob, agent, env)
            action_reward = 0
            for _ in xrange(frame_skip):
                if done:
                    break
                show()
                next_ob, reward, done, _ = env.step(action)
                action_reward += reward

            death_reward = -1 if done else 0
            rb.add_experience(ob, next_ob, action, action_reward + death_reward)
            ob = next_ob

            if rb.filled():
                agent.learn(rb.sample())
        print("Reward in episode {0}: {1}".format(episode, total_reward))
    env.monitor.close()
