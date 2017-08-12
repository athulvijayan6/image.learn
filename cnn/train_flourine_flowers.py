import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import time
import sys, os, json
from simple_cnn_model import SimpleCNN

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']

sys.path.append(os.path.join(AI_HOME, 'image'))
sys.path.append(os.path.join(AI_HOME))
from preprocessing import vgg_preprocessing
from datasets import dataset_utils, flowers

data_dir = os.path.join(AI_DATA, 'flowers', 'dataset')
train_dir = os.path.join(AI_DATA, 'flowers', 'mininet_model')

def create_dataset():
    url = "http://download.tensorflow.org/data/flowers.tar.gz"
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)

    dataset_utils.download_and_uncompress_tarball(url, data_dir)


if __name__=='__main__':
    graph = tf.Graph()
    with tf.Session(graph= graph) as session:
        dataset = flowers.get_split('train', data_dir)
        # create_dataset()
        vgg = SimpleCNN(graph= graph, session= session, train_dir= train_dir)
        vgg.train(dataset, max_steps = 3)
