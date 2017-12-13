import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from HeliumTrainer import Helium
from datasets.preprocessing import vgg_preprocessing

plt.style.use('ggplot')

layers = tf.contrib.layers
losses = tf.contrib.losses
slim = tf.contrib.slim

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']
sys.path.append(os.path.join(AI_HOME))


class Fluorine(Helium):
    def __init__(self, graph=tf.Graph(), session=None, train_dir='/tmp/simple_cnn/train'):
        super(Fluorine, self).__init__(graph, session, train_dir)

    # ****************************** Use custom implementation here ************************

    def model(self, images, num_classes, is_training=False):
        with slim.arg_scope([layers.conv2d],
                            padding='SAME',
                            weights_regularizer=layers.l2_regularizer(0.001)):
            with slim.arg_scope([layers.max_pool2d], kernel_size=[2, 2], stride=2):
                print(images.get_shape())
                net = layers.repeat(images, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv2')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv3')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv4')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv6')
                print(net.get_shape())
                net = layers.max_pool2d(net)
                print(net.get_shape())
                net = layers.flatten(net)
                print(net.get_shape())
                net = layers.fully_connected(net, 4096, scope='fc1')
                print(net.get_shape())
                net = layers.dropout(net, 0.5, is_training=is_training, scope='dropout1')
                print(net.get_shape())
                net = layers.fully_connected(net, 4096, scope='fc2')
                print(net.get_shape())
                net = layers.dropout(net, 0.5, is_training=is_training, scope='dropout2')
                print(net.get_shape())
                net = layers.fully_connected(net, num_classes, activation_fn=None, scope='fc3')
                print(net.get_shape())
                return net

    def load_batch(self,
                   dataset,
                   batch_size=32,
                   height=256,
                   width=256,
                   is_training=False):
        # TODO rewrite this without slim
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=8)
        image_raw, label = data_provider.get(['image', 'label'])

        image = vgg_preprocessing.preprocess_image(image_raw,
                                                   height,
                                                   width,
                                                   is_training=is_training)

        # Preprocess the image for display purposes.
        image_raw = tf.expand_dims(image_raw, 0)
        image_raw = tf.image.resize_images(image_raw, [height, width])
        image_raw = tf.squeeze(image_raw)

        images, images_raw, labels = tf.train.batch([image, image_raw, label],
                                                    batch_size=batch_size,
                                                    num_threads=1,
                                                    capacity=2 * batch_size)
        return images, images_raw, labels

    def losses(self, onehot_labels, logits):
        return tf.losses.softmax_cross_entropy(onehot_labels, logits)

    def evaluate(self, dataset, checkpoint_path=None, batch_size=32):
        # TODO
        with self.graph.as_default():
            print("Evaluating the model with a trained checkpoint.")
            if not checkpoint_path:
                checkpoint_path = tf.train.latest_checkpoint(self.train_dir)

            images, _, labels = self.load_batch(dataset, batch_size=batch_size, is_training=False)

            logits = self.model(images,
                                num_classes=dataset.num_classes,
                                is_training=False)
            predictions = tf.argmax(input=logits, axis=1)
            # Initialize variables to test
            self.session.run(tf.global_variables_initializer())

            predictions_ = predictions.eval()
            print(predictions_)


if __name__ == "__main__":
    pass
