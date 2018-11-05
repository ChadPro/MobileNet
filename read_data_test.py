# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import os
import pickle
import numpy as np
import math
from PIL import Image
import tensorflow as tf
import sys
import time
from nets import nets_factory
from datasets import dataset_factory
import cv2



mobile_dataset = dataset_factory.get_dataset('imagenet_300')
input_X, input_Y, testtest = mobile_dataset.inputs('train', 8, None)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    images = sess.run(input_X)
    
    
    coord.request_stop()
    coord.join(threads)