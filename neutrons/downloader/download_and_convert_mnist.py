# @Author: athul
# @Date:   2017-06-25T08:36:40+05:30
# @Last modified by:   athul
# @Last modified time: 2017-06-26T09:05:29+05:30

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

def _int64_feature(value):
    return tf.train.Feature(int64_list= tf.train.Int64List(value= [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list= tf.train.BytesList(value= [value]))

def add_to_tfrecord(data_set, filename):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                    (images.shape[0], num_examples))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    height = images.shape[1]
    width = images.shape[2]
    depth = images.shape[3]

    for i in range(num_examples):
        image_raw = images[i].tostring()
        feature = {'height' : _int64_feature(height),
                   'width'  : _int64_feature(width),
                   'depth'  : _int64_feature(depth),
                   'label'  : _int64_feature(int(labels[i])),
                   'image_raw' : _bytes_feature(image_raw) }
        example = tf.train.Example(features= tf.train.Features(feature= feature))
        writer.write(example.SerializeToString())
    print('Done writing tfrecords. ', filename)
    writer.close()

def run(download_dir, validation_size= 5000):
    data_sets = mnist.read_data_sets(download_dir,
                                     dtype= tf.uint8,
                                     reshape= False,
                                     validation_size= validation_size)
    add_to_tfrecord(data_sets.train, os.path.join(download_dir, 'train.tfrecords'))
    add_to_tfrecord(data_sets.validation, os.path.join(download_dir, 'validate.tfrecords'))
    add_to_tfrecord(data_sets.test, os.path.join(download_dir, 'test.tfrecords'))
