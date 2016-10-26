import gym
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import losses, slim

obs_shape = [120, 160, 3]
num_actions = 8

def random_obs():
    return np.random.random([3]+state_shape)

def make_network(obs_shape, num_actions, sess,
                 num_filters=[16,32,32,64,64],
                 num_units = [128,128]):
    """
    Constructs a q network with num_filters convolutions for
    initial layers, followed by hidden layers with units specified
    in num_units.

    obs and q_targets are placeholders,
    and q_values are the network outputs for a given state.
    """
    net = obs = tf.placeholder(tf.float32, shape=[None]+obs_shape)
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        kernel_size = 3,
                        stride=2,
                        normalizer_fn = slim.batch_norm,
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
        q_values = slim.fully_connected(net, num_actions)

    q_targets = tf.placeholder(tf.float32, [None, num_actions])
    loss = tf.reduce_mean((q_targets-q_values)**2)
    # TODO use placeholder to enable learning rate decay for Adam
    adam = tf.train.AdamOptimizer()
    train_step = adam.minimize(loss)
    init=tf.initialize_all_variables()
    sess.run(init)
    return obs, q_targets, q_values, train_step, loss

def row_idx(A, I):
    return tf.gather_nd(tf.transpose(tf.pack([tf.to_int64(tf.range(A.get_shape()[0])), I])))

class DQAgent(object):
    def __init__(self, env, discount_factor=.99):
        num_actions = env.action_space.n
        obs_shape = list(env.observation_space.shape)

        with tf.Graph().as_default():
            self.net_input, self.q = q_network(obs_shape)
        self.sess = tf.Session()
        self.discount_factor = discount_factor

        
        
