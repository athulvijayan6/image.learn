import tensorflow as tf
import os

from imagelearn.visiondatasets.converters import cifar10_to_tfrecords


class CIFAR10(object):
    num_classes = 10
    image_width = 32
    image_height = 32
    image_depth = 3

    def __init__(self, session, data_dir='/tmp/mnist'):
        super(CIFAR10, self).__init__()
        self.data_dir = data_dir
        self.session = session
        self.train_files = [os.path.join(self.data_dir,
                                         "cifar10_tfrecords",
                                         "data_batch_" + str(i) + '.tfrecords')
                            for i in range(1, 5)]
        self.validate_files = []
        self.test_files = [os.path.join(self.data_dir,
                                        "cifar10_tfrecords",
                                        "test_batch.tfrecords")]

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
        cifar10_to_tfrecords.run(self.data_dir)

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
                self.session.run(iterator.initializer, feed_dict={files: self.test_files})
            return batch


if __name__ == "__main__":
    data_directory = os.path.join('/', 'data', 'datasets', 'cifar10')
    batch_size_ = 32
    num_epochs_ = 10
    sess = tf.Session()

    cifar10_data = CIFAR10(sess, data_directory)
    images, labels = cifar10_data.load_batch(batch_size_, num_epochs_, is_training=True)
    images_, labels_ = sess.run([images, labels])
    print("Testing: loaded batch of inputs with shape %s" % str(images_.shape))
    print("Testing: loaded batch of labels with shape %s" % str(labels_.shape))
