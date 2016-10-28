import logging
from components import learn_doom, basic_env, eps_greedy, ReplayBuffer

from dq_agent import DQAgent

def main():
    episodes = 100000
    env = basic_env()
    agent = DQAgent(env)
    actor = eps_greedy(agent.get_actor(), episodes)
    rb = ReplayBuffer(agent.get_learner(), env, 500)
    learn_doom(agent, env, actor, rb.get_learner(), episodes=episodes)
    

if __name__=='__main__':
    main()
