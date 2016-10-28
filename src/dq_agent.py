import gym
import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import losses, slim


class DQAgent(object):

    def __init__(self, env, discount_factor=.99):
        self.num_actions = env.action_space.n
        self.obs_shape = list(env.observation_space.shape)

        with tf.Graph().as_default():
            net = self.make_network()
        self.discount_factor = discount_factor

    def make_network(self, num_filters=[32, 32, 64, 64], num_units=[256, 256]):
        net = self.obs = tf.placeholder(
            tf.float32, shape=[None] + self.obs_shape)
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
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer):
            for units in num_units:
                net = slim.fully_connected(net, units)
                net = nn.relu(net)
            self.q_values = slim.fully_connected(net, self.num_actions,
                                                 weights_initializer=tf.zeros_initializer)

        self.q_targets = tf.placeholder(tf.float32, [None, self.num_actions])
        self.loss = tf.reduce_mean((self.q_targets - self.q_values)**2)
        adam = tf.train.AdamOptimizer()
        self.train_step = adam.minimize(self.loss)
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def Q(self, obs):
        q_values = self.q_values.eval({self.obs: obs}, self.sess)
        return q_values

    def learn(self, ob, ob_next, a, r):
        batch_size = len(ob)
        q_values = self.Q(ob)
        q_targets = q_values.copy()
        q_targets[np.arange(batch_size), a.astype(int)] = r + self.discount_factor * \
            np.max(self.Q(ob_next), 1)
        self.sess.run([self.loss, self.train_step], {
            self.obs: ob, self.q_targets: q_targets})

    def act(self, obs, env, episode):
        q_values = self.Q([obs])
        return np.argmax(q_values)

    def get_actor(self):
        return lambda x,y,z: self.act(x,y,z)

    def get_learner(self):
        return lambda w, x, y, z: self.learn(w, x, y, z)
