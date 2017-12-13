import struct

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
    train_data_images = [os.path.join(data_dir, "train-images-idx3-ubyte.gz")]
    train_data_labels = [os.path.join(data_dir, "train-labels-idx1-ubyte.gz")]

    test_data_images = [os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")]
    test_data_labels = [os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")]
    output_dir = os.path.join(data_dir, "mnist_tfrecords")
    image_width = 28
    image_height = 28
    image_depth = 1

    image_files = train_data_images + test_data_images
    label_files = train_data_labels + test_data_labels
    train_record_names = ["train_" + str(i) + ".tfrecords" for i in range(len(train_data_images))]
    test_record_names = ["test_" + str(i) + ".tfrecords" for i in range(len(test_data_images))]
    record_names = train_record_names + test_record_names
    for f_images, f_labels, record_name in zip(image_files, label_files, record_names):
        out_file = os.path.join(output_dir, record_name)
        if not os.path.isfile(out_file):
            writer = tf.python_io.TFRecordWriter(out_file)

            with open(f_labels, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                labels = np.fromfile(flbl, dtype=np.int8)
            with open(f_images, 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows, cols)

            assert len(images) == len(labels)
            for image, label in zip(images, labels):
                image_raw = Image.fromarray(image).tobytes()
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
    run(data_directory)
