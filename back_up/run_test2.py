# -- coding: utf-8 --

import os
import pickle
import numpy as np
import math
from PIL import Image
import tensorflow as tf
import data_17flowers
import sys
import time
import mobile_net
import cv2
#数据batch大小
BATCH_SIZE = 5
OUT_CLASS = 17

#模型保存路径及文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
LOG_PATH = "./path/to/log"

x = tf.placeholder(tf.float32, [BATCH_SIZE, mobile_net.IMAGE_SIZE, mobile_net.IMAGE_SIZE, mobile_net.NUM_CHANNELS], name='x-input')  
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_CLASS], name='y-input')
label_y_ = tf.argmax(y_, 1)


y, _, __ = mobile_net.mobile_net(x, num_classes=OUT_CLASS, is_training=True)
output_y = tf.argmax(y, 1)

input_X_val, input_Y_val, _ = data_17flowers.inputs('val', BATCH_SIZE, None)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    img_val, label_val = sess.run([input_X_val, input_Y_val])
    label, output = sess.run([label_y_, output_y], feed_dict={x:img_val, y_:label_val})
    print label
    print output

    coord.request_stop()
    coord.join(threads)