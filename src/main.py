import numpy as np

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
inshape = [120,160,3]

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Merge, Flatten
from keras.utils.np_utils import to_categorical
from keras import backend as K
num_actions = 8
net = Sequential([Conv2D(32, 6, 6, input_shape=(120, 160, 3), subsample=(3,3)),
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
                ob = ob.astype(float)
                ob/=255
                ob-=.5
                a = agent.select_action(ob)
                ob_next, r, t, info = env.step(a)
                agent.learn(ob,a,r,t)
                ob = ob_next