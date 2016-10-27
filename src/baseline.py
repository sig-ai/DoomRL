import logging
from components import learn_doom, basic_env

from dq_agent import DQAgent

def main():
    env = basic_env()
    agent = DQAgent(env)
    learn_doom(agent, env, episodes=10, replay_buffer_size=3)
    

if __name__=='__main__':
    main()
