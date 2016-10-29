from __future__ import division
import numpy as np
import time

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
        logdir='/tmp/doom-agent-results'):
    """
    Trains using the actor function and learner function in specified env.

    episodes: the number of episodes to run.
    """

    env.monitor.start(logdir, force=True, seed=0)
    print("Outputting results to {0}".format(logdir))

    show = lambda: env.render() if render else None
    
    actor_time = 0
    step_time = 0
    learn_time = 0
    execution_start = time.time()
    counter = 0

    for episode in xrange(episodes):
        print("Episode {0}\n".format(episode))

        done = False     # Whether the current episode concluded
        prev_ob = None   # The previous observation
        ob = env.reset()  # The current observation
        total_reward = 0
        while not done:
            start = time.time()
            action = actor(ob, env, episode)
            actor_time += time.time() - start
            action_reward = 0
            for _ in xrange(frame_skip):
                if done:
                    break
                show()
                start = time.time()
                next_ob, reward, done, _ = env.step(action)
                step_time += time.time() - start
                action_reward += reward

            death_reward = -1 if done else 0
            start = time.time()
            learner(ob, next_ob, action, action_reward + death_reward)
            learn_time += time.time() - start
            counter += 1
            if counter % 50 == 0:
                print("Actor Time", actor_time)
                print("Step Time", step_time)
                print("Learn Time", learn_time)
                print("Total Time", time.time() - execution_start)
                print("")
            ob = next_ob

        print("Reward in episode {0}: {1}".format(episode, total_reward))
    env.monitor.close()
