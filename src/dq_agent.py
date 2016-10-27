import gym
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import losses, slim

obs_shape = [120, 160, 3]
num_actions = 8

import numpy as np
from random import randint


def random_obs(n):
    return np.random.random([n] + obs_shape)


def random_batch(n):
    ob = random_obs(n)
    ob_next = random_obs(n)
    r = np.random.random(n)
    a = np.array([randint(0, 8) for _ in range(n)])
    return ob, ob_next, a, r


def make_network(obs_shape, num_actions, sess,
                 num_filters=[16, 32, 32, 64, 64],
                 num_units=[128, 128]):
    """
    Constructs a q network with num_filters convolutions for
    initial layers, followed by hidden layers with units specified
    in num_units.

    obs and q_targets are placeholders,
    and q_values are the network outputs for a given state.
    """
    return obs, q_targets, q_values, train_step, loss


class DQAgent(object):

    def __init__(self, env, discount_factor=.99):
        num_actions = env.action_space.n
        obs_shape = list(env.observation_space.shape)

        with tf.Graph().as_default():
            net = self.make_network()
        self.discount_factor = discount_factor

    def make_network(self, num_filters=[32, 32, 64, 64], num_units=[256, 256]):
        net = self.obs = tf.placeholder(tf.float32, shape=[None] + obs_shape)
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            kernel_size=3,
                            stride=2,
                            normalizer_fn=slim.batch_norm,
                            activation_fn=nn.relu,
                            weights_initializer=slim.xavier_initializer_conv2d()):
            for num_outputs in num_filters:
                net = slim.conv2d(net, num_outputs)
        net = slim.flatten(net)
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer()):
            for units in num_units:
                net = slim.fully_connected(net, units)
                net = nn.relu(net)
            self.q_values = slim.fully_connected(net, num_actions)

        self.q_targets = tf.placeholder(tf.float32, [None, num_actions])
        self.loss = tf.reduce_mean((self.q_targets - self.q_values)**2)
        adam = tf.train.AdamOptimizer()
        self.train_step = adam.minimize(self.loss)
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def Q(self, obs):
        q_values = self.q_values.eval({self.obs: obs}, self.sess)
        return q_values

    def learn(self, batch):
        batch_size = len(batch[0])
        ob, ob_next, a, r = batch
        q_values = Q(self, ob)
        q_targets = q_values.copy()
        q_targets[np.arange(batch_size), a] = r + self.discount_factor * \
            np.max(Q(self, ob_next), 1)
        self.sess.run([self.loss, self.train_step], {
            self.obs: ob, self.q_targets: q_targets})

    def act(self, obs):
        q_values = Q(self, [obs])
        return np.argmax(q_values)


def Q(agent, obs):
    q_values = agent.q_values.eval({agent.obs: obs}, agent.sess)
    return q_values


def learn(agent, batch):
    batch_size = len(batch[0])
    ob, ob_next, a, r = batch
    q_values = Q(agent, ob)
    q_targets = q_values.copy()
    q_targets[np.arange(batch_size), a] = r + agent.discount_factor * \
        np.max(Q(agent, ob_next), 1)
    agent.sess.run([agent.loss, agent.train_step], {
        agent.obs: ob, agent.q_targets: q_targets})


def act(agent, obs):
    q_values = Q(agent, [obs])
    return np.argmax(q_values)
