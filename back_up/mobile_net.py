# -- coding: utf-8 --

import tensorflow as tf

import numpy as np
from collections import namedtuple

#输入与输出层
IMAGE_SIZE = 224
NUM_CHANNELS = 3
STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 17
ACTIVATION = tf.nn.relu

def separable_conv2d(inputs, dw_size, pw_size, downsample=False, is_training=True, scope=''):
    _stride = [1,2,2,1] if downsample else [1,1,1,1]
    with tf.variable_scope(scope):
        dw_filter = tf.get_variable("dw", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        dw_net = tf.nn.depthwise_conv2d(inputs, dw_filter, _stride, padding='SAME', name='dw_net', data_format='NHWC')
        dw_bn = tf.contrib.layers.batch_norm(dw_net, decay=0.7, center=True, scale=True, is_training=is_training, scope='dw_bn')
        dw_active = ACTIVATION(dw_bn)

        pw_filter = tf.get_variable("pw", pw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        pw_net = tf.nn.conv2d(dw_active, pw_filter, strides=[1,1,1,1], padding='SAME')
        pw_bn = tf.contrib.layers.batch_norm(pw_net, decay=0.7, center=True, scale=True, is_training=is_training, scope='pw_bn')
        pw_active = ACTIVATION(pw_bn)
    return pw_active, dw_filter, pw_filter

# def separable_conv2d(inputs, dw_size, pw_size, downsample=False, is_training=True,scope=''):
#     stride = [1,2,2,1] if downsample else [1,1,1,1]
        
#     with tf.variable_scope(scope):
#         depthwise_filter = tf.get_variable("dw", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
#         pointwise_filter = tf.get_variable("pw", pw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
#         separable_conv = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, stride, padding='SAME')

#         # mean, var = tf.nn.moments(separable_conv, axes=[0, 1, 2])
#         # inputs_shape = separable_conv.get_shape()
#         # params_shape = inputs_shape[-1:]
#         # scale = tf.get_variable("scale", params_shape,initializer=tf.ones_initializer())
#         # offset = tf.get_variable("offset", params_shape, initializer=tf.zeros_initializer())
#         # epsilon = 0.001
#         # bn = tf.nn.batch_normalization(separable_conv, mean, var, offset, scale, epsilon)

#         bn = tf.contrib.layers.batch_norm(separable_conv, center=True, scale=True, is_training=is_training, scope='bn')
#         net = ACTIVATION(bn)

#     return net, depthwise_filter, pointwise_filter

def mobile_net(inputs, \
                num_classes=DEFAULT_OUTPUT_NODE, \
                is_training=True, \
                reuse=None, \
                scope='mobile_net'):

    nets_dict = {}
    weights_dict = {}
    # conv
    with tf.variable_scope('conv_1'):
        conv1_weights = tf.get_variable("weight", [3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d(inputs, conv1_weights, strides=[1,2,2,1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
    nets_dict['conv_1'] = net
    weights_dict['conv_1'] = conv1_weights

    # separable net ---- conv_2~7
    net, dw_2, pw_2 = separable_conv2d(net, [3, 3, 32, 1], [1, 1, 32, 64], is_training=is_training, scope = 'conv_2')
    nets_dict['conv_2'] = net
    weights_dict['conv_2'] = dw_2
    net, dw_3, pw_3 = separable_conv2d(net, [3, 3, 64, 1], [1, 1, 64, 128], downsample=True, is_training=is_training, scope='conv_3')
    net, dw_4, pw_4 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 128], is_training=is_training, scope='conv_4')
    net, dw_5, pw_5 = separable_conv2d(net, [3, 3, 128, 1], [1, 1, 128, 256], downsample=True, is_training=is_training, scope='conv_5')
    net, dw_6, pw_6 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 256], is_training=is_training, scope='conv_6')
    net, dw_7, pw_7 = separable_conv2d(net, [3, 3, 256, 1], [1, 1, 256, 512], downsample=True, is_training=is_training, scope='conv_7')
    nets_dict['conv_7'] = net
    weights_dict['conv_7'] = dw_7

    # separable net * 5 ----conv_8
    '''
    net = depthwise_separable_conv(net, [3, 3, 512, 1], [1, 1, 512, 512], scope='conv_8_1')
    net = depthwise_separable_conv(net, [3, 3, 512, 1], [1, 1, 512, 512], scope='conv_8_2')
    net = depthwise_separable_conv(net, [3, 3, 512, 1], [1, 1, 512, 512], scope='conv_8_3')
    net = depthwise_separable_conv(net, [3, 3, 512, 1], [1, 1, 512, 512], scope='conv_8_4')
    net = depthwise_separable_conv(net, [3, 3, 512, 1], [1, 1, 512, 512], scope='conv_8_5')
    '''

    # separable net ---- conv_9~10
    net, dw_9, pw_9 = separable_conv2d(net, [3, 3, 512, 1], [1, 1, 512, 1024], downsample=True, is_training=is_training, scope='conv_9')
    net, dw_10, pw_10 = separable_conv2d(net, [3, 3, 1024, 1], [1, 1, 1024, 1024], is_training=is_training, scope='conv_10')
    
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

