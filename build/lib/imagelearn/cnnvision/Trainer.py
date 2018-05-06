import abc

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from datetime import datetime
import time


class Trainer(object):
    """
    This example implements a stable training and evaluating framework for Deep Learning.
    This is a production friendly implementation for reuse and deployment.

    This is implemented to train a net with multiple GPU towers and CPU coordinating the training process.

    The ai.universe base library for images (Neon) implements following features.
        1.  --> Done. Checkpoint training - ability to save and restore from check points.
        3.  --> Done. Summary is saved every 100 steps.
        4.  --> Done. Checkpoint is saved at every 1000 steps.
        5.  --> Done. Optional parameter in train to restore the model from checkpoint.
        6.  --> Done. Mini batch loader.abstracted
        7.  --> Done(Optional) Superviser based. For stable training in AWS.
        8.  --> TODO Evaluation metrics.
        9.  --> Done. Distributes training batch across GPUs.
        10. --> Done. Exponential learning rate decay as per --
        11. --> Done Alternative optimizers
        11. --> TODO Variable initializer according to --
        12. --> TODO Mini batch normalization.
        13. --> TODO Visualize function- creates plots and visualizations.
        14. --> TODO user API.

    The idea is to reuse the implementation for other models just by changing the model definition.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, session, data_set, train_dir):
        super(Trainer, self).__init__()
        self.train_dir = train_dir
        self.session = session

        self.TOWER_NAME = 'image-learn-tower'
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.data_set = data_set
        self.global_step = None

        self._has_built_graph = False
        self._train_op = None
        self._summary_op = None
        self._total_loss = None
        self._batch_size = None
        self._num_epochs = None

    # -------------------------------- Abstract methods --------------------------------
    @abc.abstractmethod
    def optimizer(self):
        """
        abstract method for defining optimizer
        :return:
        """
        optimizer = tf.train.AdamOptimizer()
        return optimizer

    @abc.abstractmethod
    def load_batch(self, batch_size, num_epochs, is_training):
        """
        abstract method for loading a batch of data
        :param batch_size:
        :param is_training:
        :param num_epochs:
        :return:
        """
        images = np.array([])
        labels = np.array([])
        return images, labels

    @abc.abstractmethod
    def model(self, input_data, is_training):
        """
        abstract method for model definition
        :param input_data:
        :param is_training:
        :return:
        """
        return

    @abc.abstractmethod
    def losses(self, targets, logits):
        """
        abstract method for defining loss function
        :param targets:
        :param logits:
        :return:
        """
        return

    @abc.abstractmethod
    def evaluate(self, batch_size, checkpoint_path=None):
        """
        Evaluates and computes the performance metrics for the model
        :param batch_size:
        :param checkpoint_path:
        :return:
        """
        return

    # ---------------------------------------------------------------------------------
    def build_graph(self, batch_size, num_epochs):
        if self._has_built_graph:
            return
        num_gpu = len(self._list_available_gpu())
        if num_gpu > 1:
            unpack = self._build_graph_multi_gpu(batch_size=batch_size, num_epochs=num_epochs, num_gpu=num_gpu)
        else:
            unpack = self._build_graph_simple(batch_size=batch_size, num_epochs=num_epochs)
        self._train_op = unpack[0]
        self._summary_op = unpack[1]
        self._total_loss = unpack[2]
        self.global_step = unpack[3]
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._has_built_graph = True

    def train(self):
        assert self._has_built_graph
        num_gpu = len(self._list_available_gpu())
        with self.session.graph.as_default(), tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter(self.train_dir, self.session.graph)
            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())

            # Start queues to fetch data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            # Execute the training process
            trainer_msg = "Training %s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
            num_examples_per_step = self._batch_size * num_gpu
            num_steps_per_epoch = int(self.data_set.num_train_examples / num_examples_per_step)
            print("Training started at : %s" % (datetime.now()))
            while True:
                try:
                    if coord.should_stop():
                        print("Got signal from coordinator to stop straining")
                        break
                    step_start_time = time.time()
                    _, step_loss, step = self.session.run([self._train_op, self._total_loss, self.global_step])
                    duration = time.time() - step_start_time
                    assert not np.isnan(step_loss), "Training diverged with loss = NaN at step %s" % step
                    # Verbose based of step count
                    if step % num_steps_per_epoch == 0:
                        # Just print status
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / num_gpu
                        print(trainer_msg % (datetime.now(), step, step_loss, examples_per_sec, sec_per_batch))
                    if step % (10 * num_steps_per_epoch):
                        # Write summaries to file
                        summary = self.session.run(self._summary_op)
                        summary_writer.add_summary(summary, step)
                    if step % (100 * num_steps_per_epoch):
                        # Save the model to checkpoint
                        checkpoint_path = os.path.join(self.train_dir, 'latest_model.ckpt')
                        saver.save(self.session, checkpoint_path, global_step=self.global_step)
                except tf.errors.OutOfRangeError:
                    print('Done training - epoch limit reached.')
                    coord.request_stop()
                    break
            coord.join(threads)

    def restore(self, checkpoint_path, batch_size, num_epochs):
        # restore model from a saved checkpoint
        with self.session.graph.as_default():
            images, labels = self.load_batch(batch_size=batch_size, num_epochs=num_epochs, is_training=False)
            self.model(images, is_training=False)
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                print("Evaluating the model with a trained checkpoint.")
                print(checkpoint.model_checkpoint_path)
                saver.restore(self.session, checkpoint.model_checkpoint_path)
            else:
                print("No valid checkpoint found")
                return

    # TODO
    def deploy(self):
        pass

    # TODO
    def inference(self, test_data):
        # takes a test data and create model performance summary
        pass

    # TODO
    def analyze_training(self):
        # Analyze the optimization path during the training process
        pass

    @staticmethod
    def _list_available_gpu():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def _build_graph_multi_gpu(self, batch_size, num_epochs, num_gpu):
        print('Starting training. Trained model will save model to %s' % self.train_dir)

        # Use cpu0 as the coordinator device.
        # CPU will act as a master and distribute training tasks to the slave GPUs
        with self.session.graph.as_default(), tf.device('/cpu:0'):
            # Each GPU is given a batch to compute gradient
            # The gradients from each GPU is collected by master CPU to update the weights
            # GPUs get synchronized at end of each batch. (or a set of mini batches)
            # Number of steps to train call = num_batch_processed * num_gpu
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)

            optimizer = self.optimizer()
            # Distribute training across multiple gpu
            tower_grads = []
            tower_losses = []
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.variable_scope("%s_%d" % (self.TOWER_NAME, i), reuse=True) as scope:
                        # Get a batch of data in dimension [batch_size, d0, d1,...,dn]
                        images, labels = self.load_batch(batch_size=batch_size, num_epochs=num_epochs, is_training=True)
                        # get the logits from the model definition
                        logits = self.model(images, is_training=True)

                        # Specify the loss function
                        one_hot_labels = tf.one_hot(labels, self.data_set.num_classes)
                        cross_entropy = self.losses(one_hot_labels, logits)
                        tf.add_to_collection('losses', cross_entropy)

                        # Collect the losses
                        losses = tf.get_collection('losses', scope.name)
                        total_loss = tf.add_n(losses, name='total_loss')
                        tower_losses.append(total_loss)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)
            tower_loss = tf.reduce_mean(tf.stack(tower_losses))
            # Synchronization point
            average_grads = []
            for grads_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grads_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, axis=0)
                v = grads_and_vars[0][1]
                average_grads.append((grad, v))

            # Define optimization step
            optimizer_step = optimizer.apply_gradients(average_grads, global_step=global_step)

            # add ops for summaries
            tf.summary.scalar("tower_loss", tower_loss)
            summary_op = tf.summary.merge_all()
            train_op = tf.group(optimizer_step)
            return [train_op, summary_op, tower_loss, global_step]

    def _build_graph_simple(self, batch_size, num_epochs):
        print('Starting training. Trained model will save model to %s' % self.train_dir)

        with self.session.graph.as_default(), tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)

            optimizer = self.optimizer()
            # Get a batch of data in dimension [batch_size, d0, d1,...,dn]
            images, labels = self.load_batch(batch_size=batch_size, num_epochs=num_epochs, is_training=True)
            # get the logits from the model definition
            logits = self.model(images, is_training=True)

            # Specify the loss function
            one_hot_labels = tf.one_hot(labels, self.data_set.num_classes)
            cross_entropy = self.losses(one_hot_labels, logits)
            # Define optimization step
            optimizer_step = optimizer.minimize(cross_entropy, global_step=global_step)

            # add ops for summaries
            tf.summary.scalar("cross_entropy", cross_entropy)
            summary_op = tf.summary.merge_all()
            train_op = tf.group(optimizer_step)
            return [train_op, summary_op, cross_entropy, global_step]
