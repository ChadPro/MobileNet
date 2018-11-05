# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from nets import nets_factory

tf.app.flags.DEFINE_string('net_chose','mobile_net_224_original', 'Use to chose net.')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string('restore_model_dir', './model/model.ckpt', 'Directory where checkpoints are written to.')
FLAGS = tf.app.flags.FLAGS

def export_graph(model_name):
    graph = tf.Graph()
    with graph.as_default():
        input_image = tf.placeholder(tf.float32, shape=[None,224,224], name='inputdata')
        mobile_net = nets_factory.get_network(FLAGS.net_chose)
        logits, _, __ = mobile_net.mobile_net(x, num_classes=FLAGS.num_classes, is_training=False)
        y_conv = tf.nn.softmax(logits, name='outputdata')
        restore_saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        restore_saver.restore(sess, FLAGS.restore_model_dir)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['outputdata'])

        with tf.gfile.GFile('./model.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())

export_graph('model,pb')

