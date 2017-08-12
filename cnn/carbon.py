# @Author: athul
# @Date:   2017-06-14T05:36:17+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-16T21:58:05+05:30



import copy
from datetime import datetime
import os.path
import re, sys, os
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
losses = tf.contrib.losses
metrics = tf.metrics
arg_scope = tf.contrib.framework.arg_scope

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']
sys.path.append(os.path.join(AI_HOME))
from image.helium import Helium

class Carbon(Helium):

    def __init__(self, neutron, graph= tf.Graph(), session= None, train_dir= '/tmp/beryllium/train'):
        super(Carbon, self).__init__(graph, session, train_dir)
        self.neutron = neutron
        # Optional override of hyperparameters
        # super(Carbon, self).RMSPROP_DECAY = 0.9                # Decay term for RMSProp.

    def model(self, images, num_classes, is_training= False):
        with arg_scope([layers.conv2d],
                            padding='SAME',
                            weights_regularizer= layers.l2_regularizer(0.001)):
            with arg_scope([layers.max_pool2d], kernel_size= [2, 2], stride= 2):
                print(images.get_shape())
                net = layers.repeat(images, 2, layers.conv2d, 24, [3, 3], scope= 'conv1')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 2, layers.conv2d, 12, [3, 3], scope='conv2')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.flatten(net)
                print(net.get_shape())
                net = layers.fully_connected(net, 32, scope='fc1')
                print(net.get_shape())
                net = layers.dropout(net, 0.5, is_training= is_training, scope='dropout1')
                print(net.get_shape())
                net = layers.fully_connected(net, num_classes, activation_fn= None, scope='fc3')
                print(net.get_shape())
                return net

    def load_batch(self, batch_size= 32, is_training= False):
        return self.neutron.load_batch(batch_size= batch_size, is_training= is_training)

    def evaluate(self, checkpoint_dir, checkpoint_name= None, batch_size= 32, max_steps = 100):
        with self.graph.as_default():
            images, labels = self.load_batch(batch_size= batch_size, is_training= False )
            logits = self.model(images,
                                num_classes= self.neutron.num_classes,
                                is_training= False)

            predictions = tf.argmax(input= logits, axis= 1)
            predictions = tf.cast(predictions, labels.dtype)
            _, accuracy_op = metrics.accuracy(labels, predictions)

            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

            saver = tf.train.Saver(variables_to_restore)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(checkpoint_dir, self.graph)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir, checkpoint_name)
            if ckpt and ckpt.model_checkpoint_path:
                print("Evaluating the model with a trained checkpoint.")
                print(ckpt.model_checkpoint_path)
                saver.restore(self.session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint found")
                return

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord= coord)
            try:
                for step in range(max_steps):
                    _accuracy = self.session.run([accuracy_op])
                    print("Evaluation accuracy: ", _accuracy)

            except tf.errors.OutOfRangeError as e:
                print("evaluation done")
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__=="__main__":
    pass
