# @Author: athul
# @Date:   2017-06-07T19:37:34+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-17T20:04:55+05:30



import tensorflow as tf
import numpy as np
import time
import sys, os, json
import matplotlib.pyplot as plt

from cnn.LithiumModel import LithiumModel
from datasets.mnist import MNIST

plt.style.use('ggplot')

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']

sys.path.append(AI_HOME)
sys.path.append(os.path.join(AI_HOME, 'image'))

data_dir = os.path.join(AI_DATA, 'datasets', 'mnist')
train_dir = os.path.join(AI_DATA, 'mnist', 'cnn_model')


def test_input():
    with tf.Session() as session:
        with session.graph.as_default():
            data_set = MNIST(session, data_dir)
            data_set.download_and_convert()
            # Start queues to fetch data
            batch_size = 64
            images, labels = data_set.load_batch(batch_size=batch_size, is_training=True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            images, labels = session.run([images, labels])

            i = 9
            img = np.asarray(images[i, :, :, 0])
            print(labels[i])
            plt.imshow(img)


def train():
    batch_size = 64
    num_epochs = 100
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as session:
        data_set = MNIST(session, data_dir)
        be = LithiumModel(data_set, session=session, num_epochs=num_epochs, train_dir=train_dir)
        be.train(batch_size=batch_size)
        session.close()


def evaluate():
    pass

if __name__ == '__main__':
    train()
    # evaluate()
    plt.show()
