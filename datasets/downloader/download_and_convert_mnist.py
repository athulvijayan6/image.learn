import gzip
import pathlib
import struct
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

plt.style.use('ggplot')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def run(data_dir):
    data_files = [os.path.join(data_dir, "mnist.pkl.gz")]
    output_dir = os.path.join(data_dir, "mnist_tfrecords")
    pathlib.Path(data_directory).mkdir(parents=True, exist_ok=True)
    image_width = 28
    image_height = 28
    image_depth = 1

    for f_name in data_files:
        with gzip.open(f_name, 'rb') as ff:
            data = pickle.load(ff, encoding='latin1')
        record_names = ["train.tfrecords", "validation.tfrecords", "test.tfrecords"]
        for data_set, record_name in zip(data, record_names):
            out_file = os.path.join(output_dir, record_name)
            if not os.path.isfile(out_file):
                writer = tf.python_io.TFRecordWriter(out_file)
                images = np.asarray(data_set[0])
                labels = np.asarray(data_set[1])

                assert len(images) == len(labels)
                for image, label in zip(images, labels):
                    image = image.reshape([image_width, image_height, image_depth])
                    image_raw = image.astype(np.uint8).tobytes()
                    feature = {
                        'image_raw': _bytes_feature(image_raw),
                        'label': _int64_feature(label)
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                print("TFRecords %s created and written to disk at %s" %
                      (os.path.basename(record_name) + '.tfrecords', output_dir))
            else:
                print("TFRecord %s already exist. skipping to next" % out_file)
    print("Done creating TFRecords")


if __name__ == "__main__":
    data_directory = os.path.join('/', 'data', 'datasets', 'mnist')
    pathlib.Path(data_directory).mkdir(parents=True, exist_ok=True)
    run(data_directory)
