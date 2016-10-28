import logging
from components import learn_doom, basic_env, eps_greedy, ReplayBuffer

from dq_agent import DQAgent

def main():
    EPISODES = 100000
    REPLAY_BUFFER_CAPACITY = 500
    RENDER = False

    env = basic_env()
    agent = DQAgent(env)

    actor = agent.get_actor()
    actor = eps_greedy(actor, EPISODES)

    learner = agent.get_learner()
    learner = ReplayBuffer(learner, env, REPLAY_BUFFER_CAPACITY).get_learner()

    learn_doom(agent, env, actor, learner, episodes=EPISODES, render=RENDER)
    

if __name__=='__main__':
    main()
