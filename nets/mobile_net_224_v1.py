# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
'''
本脚本 mobile_net() 实现了变形的 mobile net:
其中512深度的重复特征提取层提前，改用256深度的重复特征提取层
'''
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from collections import namedtuple

#input & output param
IMAGE_SIZE = 224
NUM_CHANNELS = 3
STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

def separable_conv2d(inputs, dw_size, pw_size, downsample=False, is_training=True, scope=''):
    _stride = [1,2,2,1] if downsample else [1,1,1,1]
    with tf.variable_scope(scope):
        dw_filter = tf.get_variable("dw", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        dw_net = tf.nn.depthwise_conv2d(inputs, dw_filter, _stride, padding='SAME', name='dw_net', data_format='NHWC')
        dw_bn = tf.contrib.layers.batch_norm(dw_net, decay=BN_DECAY, center=True, scale=True, is_training=is_training, scope='dw_bn')
        dw_active = ACTIVATION(dw_bn)

        pw_filter = tf.get_variable("pw", pw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        pw_net = tf.nn.conv2d(dw_active, pw_filter, strides=[1,1,1,1], padding='SAME')
        pw_bn = tf.contrib.layers.batch_norm(pw_net, decay=BN_DECAY, center=True, scale=True, is_training=is_training, scope='pw_bn')
        pw_active = ACTIVATION(pw_bn)
    return pw_active, dw_filter, pw_filter


def mobile_net(inputs, \
                num_classes=DEFAULT_OUTPUT_NODE, \
                is_training=True, \
                reuse=None, \
                scope='mobile_net_224_v1'):

    # rgb --> bgr 
    rgb_scaled = inputs
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2],])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    nets_dict = {}
    weights_dict = {}
    # conv
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_1'):
            conv1_weights = tf.get_variable("weight", [3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            net = tf.nn.conv2d(inputs, conv1_weights, strides=[1,2,2,1], padding='SAME')
            net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
        nets_dict['conv_1'] = net
        weights_dict['conv_1'] = conv1_weights

        # separable net ---- conv_2~5
        net, dw_2, pw_2 = separable_conv2d(net, [3, 3, 32, 1], [1, 1, 32, 64], is_training=is_training, scope = 'conv_2')
        nets_dict['conv_2'] = net
        weights_dict['conv_2'] = dw_2
        net, dw_3, pw_3 = separable_conv2d(net, [3, 3, 64, 1], [1, 1, 64, 128], downsample=True, is_training=is_training, scope='conv_3')
        net, dw_4, pw_4 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 128], is_training=is_training, scope='conv_4')
        net, dw_5, pw_5 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 256], downsample=True, is_training=is_training, scope='conv_5')

        # separable net ---- conv_6
        net, dw_6_1, pw_6_1 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6_1')
        net, dw_6_2, pw_6_2 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6_2')
        net, dw_6_3, pw_6_3 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6_3')
        net, dw_6_4, pw_6_4 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6_4')
        net, dw_6_5, pw_6_5 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6_5')

        # separable net ---- conv_7
        net, dw_7, pw_7 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 512], downsample=True, is_training=is_training, scope='conv_7')
        nets_dict['conv_7'] = net
        weights_dict['conv_7'] = dw_7

        # separable net ---- conv_9~10
        net, dw_8, pw_8 = separable_conv2d(net, [3, 3, 512, 1], [1, 1, 512, 1024], downsample=True, is_training=is_training, scope='conv_8')
        net, dw_9, pw_9 = separable_conv2d(net, [3, 3, 1024, 1], [1, 1, 1024, 1024], is_training=is_training, scope='conv_9')
    
        # pool net
        net = tf.nn.avg_pool(net, (1, 7, 7, 1), (1, 7, 7, 1), padding='SAME', name='avg_pool')
        pool_shape = net.get_shape().as_list()
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped = tf.reshape(net, [pool_shape[0], nodes])

        # fc
        with tf.variable_scope('layer-fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, num_classes], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
            fc1_biase = tf.get_variable('biase', [num_classes], initializer=tf.constant_initializer(STDDEV))
            fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biase

    return fc1, nets_dict, weights_dict
