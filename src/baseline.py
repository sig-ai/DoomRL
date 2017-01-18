#!/usr/bin/python2

import logging
from components import learn_doom, envRunner, basic_env, eps_greedy, ReplayBuffer

from dq_agent import DQAgent
from multiprocessing import Process, Queue

def main():
    EPISODES = 1000000
    REPLAY_BUFFER_CAPACITY = 500
    RENDER = False
    LOGDIR = '/tmp/doom-agent-results'

    envQueue = Queue()
    myQueue  = Queue()

    envQueues = {"r":envQueue, "w":myQueue}
    myQueues  = {"r":myQueue, "w":envQueue}

    envProcess = Process(target=envRunner, args=(envQueues, EPISODES, RENDER, LOGDIR))
    envProcess.start()

    action_space = myQueue.get()
    observation_space = myQueue.get()
    #env = basic_env()


    agent = DQAgent(action_space, observation_space)

    actor = agent.get_actor()
    actor = eps_greedy(actor, EPISODES)

    learner = agent.get_learner()
    learner = ReplayBuffer(learner, observation_space, REPLAY_BUFFER_CAPACITY).get_learner()

    spaces = {"action":action_space, "observation":observation_space}
    learn_doom(envProcess, agent, myQueues, spaces, actor, learner, episodes=EPISODES, render=RENDER)
    

if __name__=='__main__':
    main()
