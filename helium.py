# @Author: athul
# @Date:   2017-06-14T07:24:24+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-17T20:10:01+05:30



import copy
from datetime import datetime
import os.path
import re, sys, os
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import tensorflow as tf
metrics = tf.metrics

class Helium(object):
    '''
    This example implements a stable training and evaluating framework for Deep Learning.
    It is not an API, Instead a working example of an efficient implementation of
    model desciption,training and evaluation.

    This is implemented to train a net with multiple GPU towers and CPU coordinating the training process.

    Common practice is to use single machine with multiple GPUs for training a model; hyperparameter
    selection can be done using different machines parallely. Can we do better?

    The ai.universe base library implements following features.
        1.  --> Done. Checkpoint training - ability to save and restore from check points.
        2.  --> Done. Creates a Moving Average of trainable variables.
        3.  --> Done. Summary is saved every 100 steps.
        4.  --> Done. Checkpoint is saved at every 1000 steps.
        5.  --> Done. Optional parameter in train to restore the model from checkpoint.
        6.  --> Done. Minibatch loader.abstracted
        7.  --> (Optional) Superviser based. For stable training at AWS.
        8.  --> Evaluate metrics on evaluation data.
        9.  --> Done. Distributes training batch across GPUs.
        10. --> Done. Exponential learning rate decay as per --
        11. --> Variable initializer according to --
        12. --> Minibatch normalization.
        13. --> Visualize function- creates plots and visualizations.
        14. --> ai.universe API.

    The idea is to reuse the implementaion for other models just by changing the model definition.
    '''
    def __init__(self,
                 graph= tf.Graph(),
                 session= None,
                 train_dir= '/tmp/simple_cnn/train'):
        self.train_dir = train_dir
        self.graph = graph
        self.session = session
        self.num_gpus = 1

        self.TOWER_NAME = 'universetower'

        # Constants dictating the learning rate schedule.
        self.RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
        self.RMSPROP_MOMENTUM = 0.0             # Momentum in RMSProp.
        self.RMSPROP_EPSILON = 1e-10              # Epsilon term for RMSProp.

        self.MOVING_AVERAGE_DECAY = 0.9999
        self.INITIAL_LEARNING_RATE = 1e-3
        self.NUM_EPOCHS_PER_DECAY = 350.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.1

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

    # multi gpu training
    def train(self, max_steps= 10000, batch_size= 32, restore_path= None):
        print('Will save model to %s' % self.train_dir)

        # Use cpu0 as the coordinator device.
        # CPU will act as a master and distribute training tasks to the slave GPUs
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Each batch is split int num_gpus and create minibatches.
            # Each GPU is given these minibatches to compute gradient
            # The gradients from each GPU is collected by master CPU to update the weights
            # GPUs get synchronized at end of each batch. (or a set of minibatches)
            # Number of steps to train call = num_batch_processed * num_gpus
            global_step = tf.get_variable('global_step', [],
                                          initializer= tf.constant_initializer(0),
                                          trainable= False)

            #  Learning rate exponential decay
            num_batches_per_epoch = self.neutron.num_samples_to_train / batch_size
            decay_step = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)
            learning_rate = tf.train.exponential_decay( self.INITIAL_LEARNING_RATE,
                                                        global_step,
                                                        decay_step,
                                                        self.LEARNING_RATE_DECAY_FACTOR,
                                                        staircase= True)
            # Define optimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                  decay= self.RMSPROP_DECAY,
                                                  momentum= self.RMSPROP_MOMENTUM,
                                                  epsilon= self.RMSPROP_EPSILON)
            # Distribute training across multiple gpus
            # give a batch of data to each GPU for computing gradients; cpu collects the gradient and makes updates
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/cpu:%d' % i):
                        with tf.name_scope("%s_%d" % (self.TOWER_NAME, i)) as scope:
                            # Get a batch of data in dimension [batch_size, d0, d1,...,dn]
                            images, labels = self.load_batch(batch_size= batch_size, is_training= True )
                            # get the logits from the model definition
                            logits = self.model(images,
                                                num_classes= self.neutron.num_classes,
                                                is_training= True)

                            predictions = tf.argmax(input= logits, axis= 1)
                            predictions = tf.cast(predictions, labels.dtype)
                            accuracy, accuracy_op = metrics.accuracy(labels, predictions)
                            # Specify the loss function
                            onehot_labels = tf.one_hot(labels, self.neutron.num_classes)
                            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels, logits)
                            tf.add_to_collection('losses', cross_entropy)

                            # Collect the losses
                            losses = tf.get_collection('losses', scope)
                            total_loss = tf.add_n(losses, name= 'total_loss')
                            tf.get_variable_scope().reuse_variables()

                            grads = optimizer.compute_gradients(total_loss)
                            tower_grads.append(grads)

            # Synchronization point
            print(tower_grads)
            average_grads = []
            for grads_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grads_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis= 0, values= grads)
                grad = tf.reduce_mean(grad, axis= 0)
                v = grads_and_vars[0][1]
                average_grads.append((grad,v))

            # Define optimization step
            optimizer_step = optimizer.apply_gradients(average_grads, global_step= global_step)

            # Create some summaries to visualize the training process:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('learning_rate', learning_rate))
            summaries.append(tf.summary.scalar('loss', total_loss))
            summaries.append(tf.summary.scalar('accuracy', accuracy))
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
            summary_op = tf.summary.merge(summaries)
            summary_writer = tf.summary.FileWriter(self.train_dir, self.graph)

            # create a Moving average of variables
            variables_ma = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
            moving_avg_op = variables_ma.apply(tf.trainable_variables())
            # Define the final training op
            train_op = tf.group(optimizer_step, moving_avg_op)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())

            # Check if we need to restore from pre trained checkpoint for fine tuning
            saver = tf.train.Saver(tf.global_variables())
            if restore_path:
                assert tf.gfile.Exists(restore_path)
                saver.restore(self.session, restore_path)
                print('%s : Pre-trained model restored from %s' %( datetime.now(),
                                                                  restore_path))

            # Start queues to fetch data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord= coord)
            # Execute the training process
            print("Training started at : %s" % (datetime.now()))

            try:
                for step in range(max_steps):
                    if coord.should_stop():
                        break
                    step_start_time = time.time()
                    _, step_loss = self.session.run([train_op, total_loss])
                    duration = time.time() - step_start_time

                    assert not np.isnan(step_loss), "Training diverged with loss = NaN at step %s"%(step)
                    # Verbose based of step count
                    if step % 1 == 0:
                        # Just print status
                        num_examples_per_step = batch_size * self.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / self.num_gpus
                        trainer_msg = ('Training %s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                       'sec/batch)' % (datetime.now(), step, step_loss, examples_per_sec, sec_per_batch))
                        print(trainer_msg)
                    if step % 10 == 0:
                        # Write summaries to file
                        self.session.run(accuracy_op)
                        summary_str = self.session.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    if step % 1000 == 0 or (step + 1) == max_steps:
                        # Save the model to checkpoint
                        checkpoint_path = os.path.join(self.train_dir, 'latest_model.ckpt')
                        saver.save(self.session, checkpoint_path, global_step= global_step)
            except tf.errors.OutOfRangeError:
                print('Done training - epoch limit reached.')
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
            coord.join(threads)

    def restore(self, checkpoint_dir, checkpoint_name= None):
        # retore model from a saved checkpoint
        with self.graph.as_default():
            images, labels = self.load_batch(batch_size= batch_size, is_training= False )
            logits = self.model(images,
                                num_classes= self.neutron.num_classes,
                                is_training= False)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir, checkpoint_name)
            if ckpt and ckpt.model_checkpoint_path:
                print("Evaluating the model with a trained checkpoint.")
                print(ckpt.model_checkpoint_path)
                saver.restore(self.session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint found")
                return

    # TODO
    def deploy(self):
        pass

    # TODO
    def inference(self, test_data):
        # takes a test dataset and create model performance summary
        pass

    # TODO
    def analyze_training(self):
        # Analyze the optimization path during the training process
        pass

if __name__=="__main__":
    pass
