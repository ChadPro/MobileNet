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
#数据batch大小
BATCH_SIZE = 128
  
#训练参数
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE= 0.0001   
TRAINING_STEPS = 6000    
MOVING_AVERAGE_DECAY = 0.99   
LEARNING_DECAY_STEP = 100
OUT_CLASS = 17

#模型保存路径及文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
LOG_PATH = "./path/to/log"

def train():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mobile_net.IMAGE_SIZE, mobile_net.IMAGE_SIZE, mobile_net.NUM_CHANNELS], name='x-input')  
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_CLASS], name='y-input')
    isTrainNow = tf.placeholder(tf.bool, name='isTrainNow')
    label_y_ = tf.argmax(y_, 1)
#前向传播结果y
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  
    y, _, __ = mobile_net.mobile_net(x, num_classes=OUT_CLASS, is_training=isTrainNow)
    global_step = tf.Variable(0, trainable=False)
    output_y = tf.argmax(y, 1)
#计算交叉熵，并加入正则-->损失函数loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)   
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean
#计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_DECAY_STEP,LEARNING_RATE_DECAY)  
#train_step 梯度下降(学习率，损失函数，全局步数) + BN Layer Params update op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 

#持久化
    saver = tf.train.Saver()  
  
    input_X, input_Y, testtest = data_17flowers.inputs('train', BATCH_SIZE, None)

    input_X_val, input_Y_val, _ = data_17flowers.inputs('val', BATCH_SIZE, None)

    init_variable = tf.global_variables_initializer()

    with tf.Session() as sess:  
        init_op = tf.global_variables_initializer()
        sess.run(init_op) 
  
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        startTime = time.time()
        for i in range(TRAINING_STEPS):
            
            X_input, Y_input, testy = sess.run([input_X, input_Y, testtest])
            
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x:X_input, y_:Y_input, isTrainNow:True})   

            if i%30 == 0:
                
                learn_rate_now = LEARNING_RATE_BASE * (LEARNING_RATE_DECAY**(step/LEARNING_DECAY_STEP))
                

                X_input_val, Y_input_val = sess.run([input_X_val, input_Y_val])
                
                result, outy, outy_ = sess.run([accuracy,output_y,label_y_], feed_dict={x:X_input_val, y_:Y_input_val, isTrainNow:False})
                # summary_str, result = sess.run([merged, accuracy], feed_dict={x:X_input_val, y_:Y_input_val})
                # writer.add_summary(summary_str, i)
                acc = result*100.0
                accStr = str(acc) + "%"

                print("############ step : %d ################"%step)
                print("   learning_rate = %g                    "%learn_rate_now)
                print("   lose(batch)   = %g                    "%loss_value)
                print("   accuracy      = " + accStr)
                print" output : ", outy[0:10]
                print" label : ", outy_[0:10]
                print(" ")
                print(" ")

            if i%100 == 0:
                saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME)

            
        # writer.close()

        durationTime = time.time() - startTime
        minuteTime = durationTime/60
        print "To train the MobileNet, we use %d minutes" %minuteTime

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    train()


if __name__== '__main__': 
    tf.app.run()