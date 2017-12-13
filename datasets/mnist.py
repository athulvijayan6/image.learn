# @Author: athul
# @Date:   2017-06-14T05:36:52+05:30
# @Last modified by:   athul
# @Last modified time: 2017-08-15T18:30:24+05:30

import os
import tensorflow as tf

from datasets.downloader import download_and_convert_mnist


class MNIST(object):
    """docstring for mnist_dataset."""
    num_classes = 10
    image_width = 28
    image_height = 28
    image_depth = 1

    def __init__(self, data_dir, session):
        self.data_dir = data_dir
        self.session = session
        self.train_files = [os.path.join(self.data_dir,
                                         "mnist_tfrecords",
                                         "train.tfrecords")]
        self.validate_files = [os.path.join(self.data_dir,
                                            "mnist_tfrecords",
                                            "validate.tfrecords")]
        self.test_files = [os.path.join(self.data_dir,
                                        "mnist_tfrecords",
                                        "text.tfrecords")]

    def _parse_example(self, example):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "label": tf.FixedLenFeature((), tf.int64)}
        parsed_example = tf.parse_single_example(example, features=features)
        image = tf.decode_raw(parsed_example["image_raw"], out_type=tf.uint8)
        image = tf.reshape(image, [self.image_width, self.image_height, self.image_depth])
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed_example["label"], tf.int32)
        return image, label

    def download_and_convert(self):
        download_and_convert_mnist.run(self.data_dir)

    def _distrort_image(self, image, label):
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
                data = data.map(self._distrort_image)
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