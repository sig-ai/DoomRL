from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import gym
import gym_pull
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete
from gym.wrappers import SkipWrapper

from handler import *
from threading import Thread, Event
from os import kill, getpid
import signal
import code

def basic_env(moves='constant-7', resolution='160x120'):
    """
    Sets up an environment with a discretized action space
    and lower resolution.
    """
    env = gym.make('ppaquette/DoomBasic-v0')
    env = SetResolution(resolution)(env)
    env = ToDiscrete(moves)(env)
    env = SkipWrapper(4)(env)
    return env

def getFromQueue(name, queues, inputs=()):
    if len(inputs)>0:
        for i in inputs:
            queues['w'].put(i)

    if len(name):
        return queues['r'].get()

from time import sleep

def envRunner(queues, episodes, render, logdir):
    env = basic_env()
    queues['w'].put(env.action_space)
    queues['w'].put(env.observation_space)

    REC_SIZE=100
    REC_FREQ=500
    record = np.zeros([episodes])

    env.monitor.start(logdir, force=True, seed=0)
    print("Outputting results to {0}".format(logdir))

    show = lambda: env.render() if render else None
    
    for episode in xrange(episodes):

        done = False     # Whether the current episode concluded
        prev_ob = None   # The previous observation
        ob = env.reset()  # The current observation
        total_reward = 0
        while not done:
            #actor = envQueue.get()
            action = getFromQueue("action", queues, inputs=(ob,))

            if done: break
            render = queues["r"].get()
            next_ob, reward, done, _ = env.step(action)

            death_reward = -1 if done else 0

            getFromQueue("", queues, inputs=(next_ob, reward, death_reward))
            total_reward += reward + death_reward
            ob = next_ob


        record[episode] = total_reward
        if (episode + 1) % REC_FREQ == 0:
            rrs = record[:episode+1].reshape([-1, REC_SIZE]).sum(axis=1)
            plt.cla()
            plt.clf()
            plt.title('Reward per %d' % (REC_SIZE,))
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.plot(rrs)
            plt.savefig('reward.png')

        print("Reward in episode {0}: {1}".format(episode, total_reward))
    env.monitor.close()

def listen(listenEnable, stop):
    while not stop.is_set():
        if not listenEnable[0]:
            raw_input()
            kill(getpid(), signal.SIGUSR1)
            listenEnable[0] = True

def quitter(some, thing):
    kill(getpid(), signal.SIGQUIT)

def learn_doom(envp, agent, queues, spaces, actor, learner, episodes=10000, render=False):
    """
    Trains using the actor function and learner function in specified en

    episodes: the number of episodes to run.
    """
    render = True

    corrupted = False
    listenEnable = [False]
    stop = Event()
    listener = Thread(target=listen, args=(listenEnable,stop))
    listener.start()

    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGINT, quitter)
    
    for episode in xrange(episodes):
        
        done = False     # Whether the current episode concluded
        while not done:
            ob = getFromQueue("ob", queues)

            while True:
                action = actor(ob, spaces["action"], episode)
                if not corrupted:
                    break
                else:
                    corrupted = False
            queues["w"].put(action)
            queues["w"].put(render)

            next_ob = queues["r"].get()
            action_reward = queues["r"].get()
            death_reward = queues["r"].get()
            if(death_reward<0):
                done = True
            #learner(ob, next_ob, action, action_reward + death_reward)
