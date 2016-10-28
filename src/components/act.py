from __future__ import division
import numpy as np
from random import random, sample
from common import decay_fn

def eps_greedy(input_actor, episodes, epsilon_range=(.8, .01)):
    """
    This takes in an actor function and returns another actor function.
    With probability `eps`, it will sample a random move.
    Otherwise, it defers to the input_actor
    """
    get_epsilon = decay_fn(episodes, epsilon_range)

    def actor(ob, env, episode):
        eps = get_epsilon(episode)
        if random() < eps:
            return env.action_space.sample()
        return input_actor(ob, env, episode)
    return actor
