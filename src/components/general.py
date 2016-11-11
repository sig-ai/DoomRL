from __future__ import division
import numpy as np

import gym
import gym_pull
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete

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
    env = gym.make('ppaquette/DoomDeathmatch-v0')
    env = SetResolution(resolution)(env)
    env = ToDiscrete(moves)(env)
    return env

def getFromQueue(name, queues, inputs=()):
    if len(inputs)>0:
        for i in inputs:
            queues['w'].put(i)

    if len(name):
        return queues['r'].get()

from time import sleep

def envRunner(queues, episodes, render, frame_skip, logdir):
    env = basic_env()
    queues['w'].put(env.action_space)
    queues['w'].put(env.observation_space)

    

    env.monitor.start(logdir, force=True, seed=0)
    print("Outputting results to {0}".format(logdir))

    show = lambda: env.render() if render else None
    
    for episode in xrange(episodes):
        print("Episode {0}\n".format(episode))

        done = False     # Whether the current episode concluded
        prev_ob = None   # The previous observation
        ob = env.reset()  # The current observation
        total_reward = 0
        while not done:
            print "Env:", "starting"
            #actor = envQueue.get()
            action = getFromQueue("action", queues, inputs=(ob,))
            print "Env:", "got action"

            action_reward = 0
            for _ in xrange(frame_skip):
                if done:
                    break
                render = queues["r"].get()
                print "Env:", "got render"
                #show()
                next_ob, reward, done, _ = env.step(action)
                print "Env:", "stepped"
                action_reward += reward

            death_reward = -1 if done else 0

            print "env:", "putting vars"
            getFromQueue("", queues, inputs=(next_ob, action_reward, death_reward))
            ob = next_ob

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

def learn_doom(envp, agent, queues, spaces, actor, learner, episodes=10000, render=False, frame_skip=1):
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
            print "running"
            ob = getFromQueue("ob", queues)
            print "got ob"

            while True:
                action = actor(ob, spaces["action"], episode)
                if not corrupted:
                    break
                else:
                    print "redoing actor!"
                    corrupted = False
            print "Got action"
            print "Putting action"
            queues["w"].put(action)

            for _ in xrange(frame_skip):
                print "putting render"
                queues["w"].put(render)

            print "getting ob"
            next_ob = queues["r"].get()
            print "getting a_r"
            action_reward = queues["r"].get()
            print "getting d_r"
            death_reward = queues["r"].get()
            if(death_reward<0):
                done = True
            #learner(ob, next_ob, action, action_reward + death_reward)
            print "done iter"

