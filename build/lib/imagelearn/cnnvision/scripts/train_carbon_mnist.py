# @Author: athul
# @Date:   2017-06-07T19:37:34+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-16T22:00:51+05:30
import argparse
import datetime
import logging

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from imagelearn.cnnvision.models.CarbonModel import CarbonModel
from imagelearn.visiondatasets.MNIST import MNIST

plt.style.use('ggplot')


def _init_logger(log_level, log_filename):
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    log_path = os.path.dirname(log_filename)
    log_basename = os.path.basename(log_filename)
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, log_basename))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def test_input(data_dir):
    with tf.Session() as session:
        data_set = MNIST(session=session, data_dir=data_dir)
        if not data_set.has_tfrecords():
            if not data_set.has_input_data():
                data_set.download()
            data_set.write_tfrecords()
        # Start queues to fetch data
        batch_size = 32
        images, labels = data_set.load_batch(batch_size=batch_size, is_training=True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        images, labels = session.run([images, labels])

        i = 9
        img = np.asarray(images[i, :, :, 0])
        print(img)
        print(labels[i])
        plt.imshow(img)
        plt.show()


def train(data_dir, train_dir, batch_size, num_epochs):
    graph = tf.Graph()
    data_set = MNIST(data_dir, graph)
    if not os.path.isdir(data_dir) or len(os.listdir(data_dir)) < 1:
        data_set.download_and_convert()

    with tf.Session(graph=graph) as session:
        c = CarbonModel(session=session, data_set=data_set, train_dir=train_dir)
        c.build_graph(batch_size=batch_size, num_epochs=num_epochs)
        c.train()
        session.close()


def evaluate():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN to train CarbonModel for MNIST data set')
    parser.add_argument('-f', '--files-dir',
                        action='store', dest='files_dir',
                        default='~/.ml/mnist/training/carbon',
                        help="Directory to keep training files")
    parser.add_argument('-d', '--data-dir',
                        action='store', dest='data_dir',
                        default='~/.ml/mnist/data',
                        help="Directory to keep data set files")
    parser.add_argument('-e', '--num-epochs', type=int,
                        action='store', dest='num_epochs',
                        default=10000,
                        help="Number of times data set will be fed for training")
    parser.add_argument('-b', '--batch-size', type=int,
                        action='store', dest='batch_size',
                        default=50,
                        help="Number of samples in a batch")
    parser.add_argument('-t', '--test-input',
                        action='store_true', dest='test_input',
                        default=False,
                        help="Perform an input test")
    parser.add_argument('-l', '--log-level', type=int,
                        action='store', dest='log_level',
                        default=1,
                        help="Log level")
    log_levels = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}
    args = parser.parse_args()
    _data_dir = args.data_dir
    _files_dir = args.files_dir
    log_file = os.path.join(_files_dir, 'log_%s.txt'.format(datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")))
    _init_logger(log_levels[args.log_level], log_file)
    if args.test_input:
        test_input(_data_dir)
    else:
        _batch_size = args.batch_size
        _num_epochs = args.num_epochs
        _train_dir = args.train_dir

        train(_data_dir, _train_dir, _batch_size, _num_epochs)
