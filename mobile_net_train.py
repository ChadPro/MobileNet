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
from py_extend import vgg_acc

#训练参数
REGULARIZATION_RATE= 0.0001    
MOVING_AVERAGE_DECAY = 0.99   

####################
#   Learn param    #
####################
tf.app.flags.DEFINE_float('learning_rate_base', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, 'Decay learning rate.')
tf.app.flags.DEFINE_integer('learning_decay_step', 500, 'Learning rate decay step.')
tf.app.flags.DEFINE_integer('total_steps', 300000, 'Train total steps.')
tf.app.flags.DEFINE_float('gpu_fraction', 0.7, 'How to use gpu.')
tf.app.flags.DEFINE_string('train_model_dir', './model/model.ckpt', 'Directory where checkpoints are written to.')
tf.app.flags.DEFINE_string('log_dir', './log_dir', 'Log file saved.')

tf.app.flags.DEFINE_string('train_data_path','', 'Dataset for train.')
tf.app.flags.DEFINE_string('val_data_path', '', 'Dataset for val.')

tf.app.flags.DEFINE_string('dataset', 'imagenet_224', 'Chose dataset in dataset_factory.')
tf.app.flags.DEFINE_bool('white_bal',False, 'If white balance.')
tf.app.flags.DEFINE_integer('image_size', 224, 'Default image size.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Default batch_size 64.')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'Number of classes to use in the dataset.')

tf.app.flags.DEFINE_string('net_chose','mobile_net_224_original', 'Use to chose net.')
tf.app.flags.DEFINE_bool('fine_tune', False, 'Is fine_tune work.')
tf.app.flags.DEFINE_string('restore_model_dir', '', 'Restore model.')
FLAGS = tf.app.flags.FLAGS 

########################################
#           Train function             #
########################################
def train():
    #1. Get mobile net
    mobile_net = nets_factory.get_network(FLAGS.net_chose)
    with tf.name_scope("Data_Input"):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], name='x-input')
        rgb_img_input = tf.reverse(x, axis=[-1])  
        tf.summary.image("input", rgb_img_input, 5)
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes], name='y-input')
        isTrainNow = tf.placeholder(tf.bool, name='isTrainNow')
        label_y_ = tf.argmax(y_, 1)
    
    #2. Forward propagation
    with tf.name_scope("Forward_Propagation"):
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  
        y, _, __ = mobile_net.mobile_net(x, num_classes=FLAGS.num_classes, is_training=isTrainNow)
        global_step = tf.Variable(0, trainable=False)
        output_y = tf.argmax(y, 1)

    #3. Calculate cross_entropy
    with tf.name_scope("Calc_Loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)   
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        loss = cross_entropy_mean

    #4. Back propagation
    with tf.name_scope("Back_Train"):
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base ,global_step, FLAGS.learning_decay_step, FLAGS.learning_rate_decay)  
        # train_step 梯度下降(学习率，损失函数，全局步数) + BN Layer Params update op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 

    #5. Calculate val accuracy
    with tf.name_scope("Calc_Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #6. Tensorboard summary and Saver persistent
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('val_acc', accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
    saver = tf.train.Saver()  

    #7. Get Dataset
    mobile_dataset = dataset_factory.get_dataset(FLAGS.dataset)
    input_X, input_Y, testtest = mobile_dataset.inputs(FLAGS.train_data_path,FLAGS.val_data_path,'train', FLAGS.batch_size, None)
    input_X_val, input_Y_val, _ = mobile_dataset.inputs(FLAGS.train_data_path,FLAGS.val_data_path,'val', FLAGS.batch_size, None)

    init_variable = tf.global_variables_initializer()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    with tf.Session(config=config) as sess:
        # init global variables  
        init_op = tf.global_variables_initializer()
        sess.run(init_op) 
  
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        if len(FLAGS.restore_model_dir) > 0:
            print "#####=============> Restore Model : "+str(FLAGS.restore_model_dir)
            saver.restore(sess, FLAGS.restore_model_dir)

        startTime = time.time()
        for i in range(FLAGS.total_steps):
            X_input, Y_input, testy = sess.run([input_X, input_Y, testtest])
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x:X_input, y_:Y_input, isTrainNow:True})   

            if i%30 == 0:     
                learn_rate_now = FLAGS.learning_rate_base * ( FLAGS.learning_rate_decay**(step/ FLAGS.learning_decay_step))
                X_input_val, Y_input_val = sess.run([input_X_val, input_Y_val])
                
                summary_str, outy, outy_ = sess.run([merged, y, y_], feed_dict={x:X_input_val, y_:Y_input_val, isTrainNow:False})
                writer.add_summary(summary_str, i)
                # acc = result*100.0
                # accStr = str(acc) + "%"
                acc_top1 = vgg_acc.acc_top1(outy, outy_)
                acc_top5 = vgg_acc.acc_top5(outy, outy_)
                run_time = time.time() - startTime
                run_time = run_time / 60

                print("############ step : %d ################"%step)
                print("   learning_rate = %g                    "%learn_rate_now)
                print("   lose(batch)   = %g                    "%loss_value)
                # print("   accuracy      = " + accStr)
                print("   acc_top1      = " + acc_top1)
                print("   acc_top5      = " + acc_top5)
                print("   train run     = %d min"%run_time)
                print(" ")
                print(" ")

            if i%500 == 0:
                saver.save(sess, FLAGS.train_model_dir)

        writer.close()
        durationTime = time.time() - startTime
        minuteTime = durationTime/60
        print "To train the MobileNet, we use %d minutes" %minuteTime
        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    train()

if __name__== '__main__': 
    tf.app.run()