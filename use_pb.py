# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import cv2
import numpy as np

tf.app.flags.DEFINE_string("img_path","./test.jpg","Test image path.")
FLAGS = tf.app.flags.FLAGS

img = cv2.imread(FLAGS.img_path)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
img = np.expand_dims(img, axis=0)

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './model.pb'

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        input_x = sess.graph.get_tensor_by_name("inputdata:0")
        output = sess.graph.get_tensor_by_name("outputdata:0")

        yy = sess.run(output, {input_x:img})
        print np.argmax(yy)

