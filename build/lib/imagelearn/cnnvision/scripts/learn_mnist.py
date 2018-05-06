import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def initialize_weights(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)

def initialize_bias(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding= "SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= "SAME")


if __name__=="__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 28*28])
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Layer 1 - convolution + ReLu + max pool
    W_convlayer_1 = initialize_weights([5, 5, 1, 32])
    b_convlayer_1 = initialize_bias([32])
    h_convlayer_1 = tf.nn.relu(conv2d(x_img, W_convlayer_1) + b_convlayer_1)
    h_pooled_1 = max_pool(h_convlayer_1)

    # Layer 2 - convolution + ReLu + max pool
    # In order to build a deep network, we stack several layers of this type.
    # The second layer will have 64 features for each 5x5 patch.

    W_convlayer_2 = initialize_weights([5, 5, 32, 64])
    b_convlayer_2 = initialize_bias([64])
    h_convlayer_2 = tf.nn.relu(conv2d(h_pooled_1, W_convlayer_2) + b_convlayer_2)
    h_pooled_2 = max_pool(h_convlayer_2)

    # Layer 3 - Fully connected layer
    W_fc_1 = initialize_weights([7 * 7 * 64, 1024])
    b_fc_1 = initialize_bias([1024])

    h_convlayer_2_flat = tf.reshape(h_pooled_2, [-1, 7 * 7 * 64])
    h_fc_1 = tf.nn.relu(tf.matmul(h_convlayer_2_flat, W_fc_1) + b_fc_1)

    # Dropout
    keep_probab = tf.placeholder(tf.float32)
    h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_probab)

    #Layer 4 - Readout layer - softmax
    W_fc_2 = initialize_weights([1024, 10])
    b_fc_2 = initialize_bias([10])
    # preprocessor to softmax
    y_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2

    #    Train
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Performance statistics
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_probab: 1.0})
            print("step %d, training accuracy %.4f"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_probab: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



