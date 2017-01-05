from __future__ import division
import numpy as np
from dq_agent import DQAgent
import gym
names=['ppaquette/DoomBasic-v0',            #0
       'ppaquette/DoomCorridor-v0',         #1
       'ppaquette/DoomDeathmatch-v0',       #2
       'ppaquette/DoomDefendCenter-v0',     #3
       'ppaquette/DoomDefendLine-v0',       #4
       'ppaquette/DoomHealthGathering-v0',  #5
       'ppaquette/DoomMyWayHome-v0',        #6
       'ppaquette/DoomPredictPosition-v0',  #7
       'ppaquette/DoomTakeCover-v0',        #8
       'ppaquette/meta-Doom-v0']            #9


def basic_env(moves='constant-7', resolution='160x120', name = 'ppaquette/DoomDeathmatch-v0'):
    """
    Sets up an environment with a discretized action space
    and lower resolution.
    """
    env = gym.make(name)
    env = SetResolution(resolution)(env)
    env = ToDiscrete(moves)(env)
    env = SkipWrapper(4)(env)
    # env = SetPlayingMode('human')(env)
    return env


na = 5
bs = 5
inshape = [84,84,1]

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Merge, Flatten
from keras.utils.np_utils import to_categorical
from keras import backend as K
num_actions = 8
def make_net(input_shape, num_actions):
    net = Sequential([Conv2D(32, 6, 6, input_shape=input_shape, subsample=(3,3)),
                  # Activation('relu'),
                  # BatchNormalization(),
                  # Conv2D(64, 6, 6, subsample=(3,3)),
                  # Activation('relu'),
                  # BatchNormalization(),
                  # Conv2D(64, 6, 6, subsample=(3,3)),
                  # Activation('relu'),
                  # BatchNormalization(),
                  Flatten(),
                  # Dense(512),
                  # Activation('relu'),
                  # BatchNormalization(),
                  # Dense(1024),
                  # Activation('relu'),
                  # BatchNormalization(),
                  Dense(num_actions)])
    return net
def learn_doom(env, agent, episodes=1, render=True):
    """
    Trains using the actor function and learner function in specified en

    episodes: the number of episodes to run.
    """
    
    for episode in xrange(episodes):
        ob = env.reset()
        for _ in xrange(episodes):
            t = False
            while not t:
                ob = scale(ob, axis=0, with_mean=True, with_std=True)
                ob/=255
                ob-=.5
                a = agent.select_action(ob)
                ob_next, r, t, info = env.step(a)
                agent.learn(ob,a,r,t)
                ob = ob_next

from skimage.color import rgb2gray
from skimage.transform import resize

from math import exp
from random import random

def learn_atari(episodes=1, render=True):
    """
    Trains using the actor function and learner function in specified en

    episodes: the number of episodes to run.
    """
    env = gym.make('Breakout-v0')
    net = make_net([84,84,2], env.action_space.n)
    agent = DQAgent(net,env.action_space.n, [84,84,2])
    for episode in xrange(episodes):
        ob = env.reset()
        ob = rgb2gray(ob)
        ob = resize(ob, [84,84])
        prev = ob
        for ep in xrange(episodes):
            print "Episode: {}".format(ep)
            explore_prob = exp(-ep/10)
            
            t = False
            total = 0
            while not t:
                s = np.stack([prev,ob],2)
                if random() < explore_prob:
                    a = env.action_space.sample()
                a = agent.select_action(s)
                ob_next, r, t, info = env.step(a)
                total+=r
                agent.learn(s,a,r,t)
                prev = ob
                ob = rgb2gray(ob_next)
                ob = resize(ob, [84,84])
            print "Reward: {}".format(total)

