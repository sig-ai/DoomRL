from __future__ import division
import numpy as np
from dq_agent import DQAgent
import gym
import tensorflow as tf
from gym.wrappers import SkipWrapper
#from components.common import decay_fn

def decay_fn(total_iterations, output_range=[1,.1]):
    """
    Returns a function that decays its outputs with respect
    to the iterations. Useful for decreasing exploration and
    learning rates over time.
    """
    first_output, last_output = output_range
    step_update = (last_output - first_output) / total_iterations

    def decay(step):
        """ Output decays linearly with iterations. """
        if step > total_iterations:
            return last_output
        return first_output + step * step_update
        
    return decay



tf.python.control_flow_ops = tf
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
    net = Sequential([Conv2D(32, 8, 8, input_shape=input_shape, subsample=(4,4)),
                  Activation('relu'),
                  BatchNormalization(),
                  Conv2D(64, 4, 4, subsample=(2,2)),
                  Activation('relu'),
                  BatchNormalization(),
                  Conv2D(64, 3, 3, subsample=(3,3)),
                  Activation('relu'),
                  BatchNormalization(),
                  Flatten(),
                  Dense(512),
                  Activation('relu'),
                  BatchNormalization(),
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
n = 6
ob_shape = [84,84,2]
game = 'Pong-v0'

def learn_atari(episodes=1, agent = None, render=True, save_steps=500,
                run_steps=100, decay=500, verbose=True):
    """
    Trains using the actor function and learner function in specified en

    episodes: the number of episodes to run.
    """
    env = gym.make(game)
    env = SkipWrapper(3)(env)
    if agent == None:
        net = make_net([84,84,2], env.action_space.n)
        agent = DQAgent(net,env.action_space.n, [84,84,2])

    steps = 0
    explore_prob = 1
    for episode in xrange(episodes):
        if episode % run_steps == 0:
            print('Test Run')
            run_atari(agent, env, render=False)
	if episode % save_steps == 0:
		print 'saving net'
		net.save('model.h5')
        ob = env.reset()
        ob = rgb2gray(ob)
        ob = resize(ob, [84,84])
        prev = ob
        print "Episode: {0}, explore_prob:{1}".format(episode, explore_prob)
        t = False
        total = 0
        while not t:
            explore_prob = decay(steps)
            steps+=1
            s = np.stack([prev,ob],2)
            if random() < explore_prob or not agent.warmed_up:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s)
            ob_next, r, t, info = env.step(a)
            total+=r
            agent.learn(s,a,r,t)
            prev = ob
            ob = rgb2gray(ob_next)
            ob = resize(ob, [84,84])
        print "Reward: {}".format(total)

def load_agent(fname = 'model.h5'):
    net = make_net([84,84,2], n)
    return DQAgent(net, n, [84,84,2])
  
def run_atari(agent=None, env = gym.make(game), eps = 5, render=True, max_steps=1000):
    if agent ==None:
	agent = load_agent()
    for _ in xrange(eps):
        t = False
        ob = env.reset()
        ob = rgb2gray(ob)
        ob = resize(ob, [84,84])
        prev = ob
        r_total = 0
        steps = 0
    	while not t and steps < max_steps:
            steps+=1
            s = np.stack([prev,ob],2)
	    if render:
                env.render()
            a = agent.select_action(s)
            ob_next, r, t, info = env.step(a)
            r_total+=r
            prev = ob
            ob = rgb2gray(ob_next)
            ob = resize(ob, [84,84])
        print "Test Reward: {}".format(r_total)
