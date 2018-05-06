# @Author: athul
# @Date:   2017-06-14T05:36:17+05:30
# @Last modified by:   athul
# @Last modified time: 2017-08-15T21:49:18+05:30

import tensorflow as tf
from imagelearn.cnnvision.Trainer import Trainer


class CarbonModel(Trainer):
    def __init__(self, session, data_set, train_dir='/tmp/beryllium/train'):
        super(CarbonModel, self).__init__(data_set, session, train_dir)

    def model(self, images, is_training=False):
        convolution_1 = tf.layers.conv2d(images,
                                         filters=32,
                                         kernel_size=32,
                                         padding='same',
                                         activation=tf.nn.relu)
        pooling_1 = tf.layers.max_pooling2d(convolution_1, pool_size=[2, 2], strides=2)
        convolution_2 = tf.layers.conv2d(pooling_1,
                                         filters=64,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         activation=tf.nn.relu)
        pooling_2 = tf.layers.max_pooling2d(convolution_2, pool_size=[2, 2], strides=2)
        pooling_2_flat = tf.layers.flatten(pooling_2)
        dense = tf.layers.dense(pooling_2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense, rate=0.4, training=is_training)
        return tf.layers.dense(dropout, units=10)

    def optimizer(self):
        learning_rate = 0.01
        return tf.train.AdamOptimizer(learning_rate)

    def losses(self, one_hot_labels, logits):
        return tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    def load_batch(self, batch_size, num_epochs, is_training=False):
        return self.data_set.load_batch(batch_size=batch_size, num_epochs=num_epochs, is_training=is_training)

    def evaluate(self, checkpoint_dir, checkpoint_name=None, batch_size=32, max_steps=100):
        with self.session.graph.as_default():
            images, labels = self.load_batch(batch_size=batch_size, is_training=False)
            logits = self.model(images, is_training=False)

            predictions = tf.argmax(input=logits, axis=1)
            predictions = tf.cast(predictions, labels.dtype)
            _, accuracy_op = metrics.accuracy(labels, predictions)

            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

            saver = tf.train.Saver(variables_to_restore)

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
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
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


if __name__ == "__main__":
    pass
