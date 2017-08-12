# @Author: athul
# @Date:   2017-06-14T05:36:52+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-15T09:37:22+05:30

import os
import tensorflow as tf
import image.neutrons.downloader.download_and_convert_mnist as downloader_mnist

class neutron_mnist():
    """docstring for mnist_dataset."""

    CLASS_NAMES = ['zero', 'one', 'two', 'three',
                        'four', 'five', 'size', 'seven', 'eight', 'nine']
    num_classes = 10
    image_width = 28
    image_height = 28
    image_depth = 1
    IMAGE_SIZE = image_width * image_height * image_depth
    TRAIN_FILE = 'train.tfrecords'
    VALIDATION_FILE = 'validate.tfrecords'
    TEST_FILE = 'test.tfrecords'
    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [28 x 28 x 1] grayscale image.',
        'label': 'A single integer between 0 and 9',
    }
    # TODO
    num_samples_to_train = 50000
    num_samples_to_test = 5000
    num_samples_to_validate = 10000

    def __init__(self, data_dir, graph, reader=None):
        self.data_dir = data_dir
        self.graph = graph
        self.reader = reader

    def load_batch(self, batch_size, is_training= True, num_threads= 1):
        # This will have batch of unprocessed inputs for eval/testing
        # This will have batch of inputs preprocessed/distorted for training
        with self.graph.as_default():
            filename = self.TRAIN_FILE if is_training else self.VALIDATION_FILE
            filenames = [os.path.join(self.data_dir, filename)]
            with tf.name_scope('neutron'):
                filename_queue = tf.train.string_input_producer(filenames)
                # get a sample
                image, label = self._read_example(filename_queue, is_training= is_training)
                # create batch of them, with shuffling
                images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                               batch_size= batch_size,
                                                               num_threads= num_threads,
                                                               capacity= 1000 + 3 * batch_size,
                                                               min_after_dequeue= 1000)
                return images, sparse_labels

    def download_and_convert(self):
        downloader_mnist.run(self.data_dir)

    def _distrort_image(self, image):
        return image

    def _read_example(self, filename_queue, is_training= True):
        # Returns an independent reader everytime reading different parts inputs.
        with self.graph.as_default():
            if self.reader is None:
                self.reader = tf.TFRecordReader()
            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,
                                               features= {'image_raw': tf.FixedLenFeature([], tf.string),
                                                          'label'    : tf.FixedLenFeature([], tf.int64 )
                                                         }
                                              )
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape([self.IMAGE_SIZE])
            image = tf.reshape(image, [self.image_height, self.image_width, self.image_depth])
            if is_training:
                image = self._distrort_image(image)

            # convert from 0-255 to -0.5 : 0.5
            image = tf.cast(image,tf.float32) * 1./255 - 0.5
            label = tf.cast(features['label'], tf.int32)

            return image, label
