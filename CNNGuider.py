import tensorflow as tf
import tensorflow.contrib.layers as tcl
import os
import tensorflow.contrib.autograph as autograph
import tensorflow.contrib.distributions as tfdist

from config import *


class Classifier(object):
    def __init__(self, name='metabayesian/d_net', class_num=CANDIDATE_NUM):
        self.name = name
        self.class_num = class_num

    def __call__(self, x, reuse=tf.AUTO_REUSE, split=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE, regularizer=tcl.l2_regularizer(1e-3)) as vs:
            x = tf.reshape(x, [-1, 200, 200, 2])
            conv0 = tcl.convolution2d(
                x, 64, [3, 3], [1, 1],
                activation_fn=tf.nn.relu
            )
            conv0 = tcl.convolution2d(
                conv0, 64, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv0 = tcl.convolution2d(
                conv0, 128, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv0 = tcl.instance_norm(conv0)
            conv1 = tcl.convolution2d(
                conv0, 128, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv1 = tcl.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv1 = tcl.convolution2d(
                conv1, 128, [3, 3], [1, 1],
                activation_fn=tf.nn.relu
            )
            conv1 = tcl.instance_norm(conv1)
            conv2 = tcl.convolution2d(
                conv1, 256, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv2 = tcl.convolution2d(
                conv2, 512, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv2 = tcl.convolution2d(
                conv2, 512, [3, 3], [1, 1],
                activation_fn=tf.nn.relu
            )
            conv6 = tcl.flatten(conv2)
            fc2 = tcl.fully_connected(
                conv6, self.class_num,
                activation_fn=tf.identity
            )
            return fc2
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


