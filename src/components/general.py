from __future__ import division
import numpy as np
from common import decay_fn

import gym
import gym_pull
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete

def basic_env(moves='constant-7', resolution='160x120'):
    """
    Sets up an environment with a discretized action space
    and lower resolution.
    """
    env = gym.make('ppaquette/DoomDeathmatch-v0')
    env = SetResolution(resolution)(env)
    env = ToDiscrete(moves)(env)
    return env


def learn_doom(agent, env, actor, learner, episodes=10000, render=False, frame_skip=1,
               logdir='/tmp/doom-agent-results',
               learning_rate_range=(.001, .0001)):
    """
    Trains the agent in specified env.

    episodes: the number of episodes to run.
    """

    env.monitor.start(logdir, force=True, seed=0)
    print("Outputting results to {0}".format(logdir))

    get_learning_rate = decay_fn(episodes, learning_rate_range)
    show = lambda: env.render() if render else None
    
    for episode in xrange(episodes):
        print("Episode {0}\n".format(episode))
        learning_rate = get_learning_rate(episode)

        done = False     # Whether the current episode concluded
        prev_ob = None   # The previous observation
        ob = env.reset()  # The current observation
        total_reward = 0
        while not done:
            action = actor(ob, env, episode)
            action_reward = 0
            for _ in xrange(frame_skip):
                if done:
                    break
                show()
                next_ob, reward, done, _ = env.step(action)
                action_reward += reward

            death_reward = -1 if done else 0
            learner(ob, next_ob, action, action_reward + death_reward)
            ob = next_ob

        print("Reward in episode {0}: {1}".format(episode, total_reward))
    env.monitor.close()
