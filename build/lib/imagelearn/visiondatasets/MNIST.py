# @Author: athul
# @Date:   2017-06-14T05:36:52+05:30
# @Last modified by:   athul
# @Last modified time: 2017-08-15T18:30:24+05:30
import logging
import os
import sys
from urllib.error import URLError

import tensorflow as tf

from imagelearn.visiondatasets.converters import mnist_to_tfrecords
from imagelearn.visiondatasets.downloaders.download_mnist import download_mnist


class MNIST(object):
    """docstring for mnist_dataset."""
    num_classes = 10
    image_width = 28
    image_height = 28
    image_depth = 1
    num_train_examples = 5000
    tfrecords = {'train': ["train.tfrecords"],
                 'validation': ["validation.tfrecords"],
                 'test': ["test.tfrecords"]}

    def __init__(self, session, data_dir):
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.session = session
        self.train_files = [os.path.join(self.data_dir,
                                         "tfrecords",
                                         i) for i in self.tfrecords['train']]
        self.validate_files = [os.path.join(self.data_dir,
                                            "tfrecords",
                                            i) for i in self.tfrecords['validation']]
        self.test_files = [os.path.join(self.data_dir,
                                        "tfrecords",
                                        i) for i in self.tfrecords['test']]
        self.input_filename = "mnist.pkl.gz"

    def has_tfrecords(self):
        for f in self.train_files + self.validate_files + self.test_files:
            if not os.path.isfile(f):
                return False
        return True

    def has_input_data(self):
        return os.path.isfile(os.path.join(self.data_dir, self.input_filename))

    def _parse_example(self, example):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "label": tf.FixedLenFeature((), tf.int64)}
        parsed_example = tf.parse_single_example(example, features=features)
        image = tf.decode_raw(parsed_example["image_raw"], out_type=tf.uint8)
        image = tf.reshape(image, [self.image_width, self.image_height, self.image_depth])
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed_example["label"], tf.int32)
        return image, label

    def download(self):
        try:
            logging.info("Attempting to download data set...")
            download_mnist(self.data_dir)
        except URLError:
            logging.warning("Could not download data set automatically. Check your internet connection.")
            sys.exit()

    def write_tfrecords(self):
        logging.info("Converting data set to tfrecords")
        data_files = [os.path.join(self.data_dir, self.input_filename)]
        mnist_to_tfrecords.run(data_files, self.data_dir, self.tfrecords)

    @staticmethod
    def _distort_image(image, label):
        # TODO add distort function
        return image, label

    def load_batch(self, batch_size, num_epochs=None, is_training=True):
        with self.session.graph.as_default():
            files = tf.placeholder(tf.string, shape=[None])
            data = tf.data.TFRecordDataset(files)
            data = data.map(self._parse_example)
            data = data.repeat(num_epochs)
            data = data.shuffle(buffer_size=10000)
            data = data.batch(batch_size)
            if is_training:
                data = data.map(self._distort_image)
            iterator = data.make_initializable_iterator()
            batch = iterator.get_next()
            if is_training:
                self.session.run(iterator.initializer, feed_dict={files: self.train_files})
            else:
                self.session.run(iterator.initializer, feed_dict={files: self.validate_files + self.test_files})
            return batch


if __name__ == "__main__":
    data_directory = os.path.join('/', 'data', 'datasets', 'mnist')
    batch_size_ = 32
    num_epochs_ = 10
    sess = tf.Session()

    mnist_data = MNIST(sess, data_directory)
    images, labels = mnist_data.load_batch(batch_size_, num_epochs_, is_training=True)
    images_, labels_ = sess.run([images, labels])
    print("Testing: loaded batch of inputs with shape %s" % str(images_.shape))
    print("Testing: loaded batch of labels with shape %s" % str(labels_.shape))
