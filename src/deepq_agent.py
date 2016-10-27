import gym
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import losses, slim


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
