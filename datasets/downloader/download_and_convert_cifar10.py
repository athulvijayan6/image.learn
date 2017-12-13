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
    train_data_files = [os.path.join(data_dir,
                                     "cifar-10-batches-py",
                                     "data_batch_" + str(i)) for i in range(1, 5)]
    test_data_files = [os.path.join(data_dir,
                                    "cifar-10-batches-py",
                                    "test_batch")]
    output_dir = os.path.join(data_dir, "cifar10_tfrecords")
    image_width = 32
    image_height = 32
    image_depth = 3

    for f_name in train_data_files + test_data_files:
        out_file = os.path.join(output_dir, os.path.basename(f_name) + '.tfrecords')
        if not os.path.isfile(out_file):
            writer = tf.python_io.TFRecordWriter(out_file)
            data = unpickle(f_name)
            images = data[b'data']
            labels = data[b'labels']

            images = np.reshape(images, (-1, image_depth, image_width, image_height))
            images = images.transpose([0, 2, 3, 1]).astype("uint8")
            assert len(images) == len(labels)

            for image, label in zip(images, labels):
                image_raw = Image.fromarray(image).tobytes()
                feature = {
                    'image_raw': _bytes_feature(image_raw),
                    'label': _int64_feature(label)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            print("TFRecords %s created and written to disk at %s" % (
            os.path.basename(f_name) + '.tfrecords', output_dir))
        else:
            print("TFRecord %s already exist. skipping to next" % out_file)
    print("Done creating TFRecords")


if __name__ == "__main__":
    data_directory = os.path.join('/', 'data', 'datasets', 'cifar10')
    run(data_directory)
