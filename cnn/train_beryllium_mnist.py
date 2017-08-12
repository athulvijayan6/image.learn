# @Author: athul
# @Date:   2017-06-07T19:37:34+05:30
# @Last modified by:   athul
# @Last modified time: 2017-07-17T20:04:55+05:30



import tensorflow as tf
import numpy as np
import time
import sys, os, json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from beryllium import Beryllium
from image.neutrons.neutron_mnist import neutron_mnist

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']

sys.path.append(AI_HOME)
sys.path.append(os.path.join(AI_HOME, 'image'))

data_dir = os.path.join(AI_DATA, 'mnist')
train_dir = os.path.join(AI_DATA, 'mnist', 'cnn_model')

def test_input():
    # create_dataset()
    graph = tf.Graph()
    neutron = neutron_mnist(data_dir, graph)
    # neutron.download_and_convert()
    with tf.Session(graph= graph) as session:
        with graph.as_default():
            # Start queues to fetch data
            batch_size = 32
            images, labels = neutron.load_batch(batch_size= batch_size, is_training= True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord= coord)
            images, labels = session.run([images, labels])

            i = 9
            img = np.asarray(images[i, :, :, 0])
            print(labels[i])
            plt.imshow(img)

def train():
    # create_dataset()
    graph = tf.Graph()
    neutron = neutron_mnist(data_dir, graph)
    # neutron.download_and_convert()
    with tf.Session(graph= graph,
                    config= tf.ConfigProto(allow_soft_placement= True, log_device_placement= False)) as session:
        be = Beryllium(neutron, graph= graph, session= session, train_dir= train_dir)
        be.train(batch_size= 64, max_steps = 1000)
        session.close()

def evaluate():
    # create_dataset()
    graph = tf.Graph()
    neutron = neutron_mnist(data_dir, graph)
    with tf.Session(graph= graph) as session:
        be = Beryllium(neutron, graph= graph, session= session, train_dir= train_dir)
        be.evaluate(train_dir)
        session.close()

if __name__ == '__main__':
    train()
    # evaluate()
    plt.show()
