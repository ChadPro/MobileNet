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
BATCH_SIZE = 64
  
OUT_CLASS = 17

#模型保存路径及文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
LOG_PATH = "./path/to/log"

def read_image():
    img1_pre = cv2.imread('image_1.jpg')
    img2_pre = cv2.imread('image_2.jpg')
    img3_pre = cv2.imread('image_3.jpg')
    img4_pre = cv2.imread('image_4.jpg')
    img5_pre = cv2.imread('image_5.jpg')
    img6_pre = cv2.imread('image_6.jpg')
    img7_pre = cv2.imread('image_7.jpg')
    img8_pre = cv2.imread('image_8.jpg')

    img1 = cv2.resize(img1_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.resize(img3_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img4 = cv2.resize(img4_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img5 = cv2.resize(img5_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img6 = cv2.resize(img6_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img7 = cv2.resize(img7_pre, (224,224), interpolation=cv2.INTER_CUBIC)
    img8 = cv2.resize(img8_pre, (224,224), interpolation=cv2.INTER_CUBIC)

    images = np.array([img3])
    return images

iii = read_image()
iiii = tf.cast(iii, tf.float32)

img_input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='x_input')

y, nets_dict, weights_dict = mobile_net.mobile_net(img_input, num_classes=OUT_CLASS, is_training=False)
ss = tf.nn.softmax(y)
output = tf.argmax(ss, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME)

    images = sess.run(iiii)

    r, conv_1, dw_7 = sess.run([output, nets_dict['conv_1'],weights_dict['conv_7']], feed_dict={img_input:images})
    print r


