import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

class PooledConvolutionLayer(object):
    """
        The model assumes:
            1. weights are shared across each units in a layer
            2. Stride = 1; We move the filter by steps of stride
        Depth of output layer is controlled by number of filters

        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
    """
    def __init__(self, input, rng, filter_shape, image_shape, poolsize= (2, 2)):

        self.input = input
        # Number of inputs to each unit
        fan_in = np.prod(filter_shape[1:])
        # receptive field size = np.prod(filter_shape[2:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize)

        # Initialize weights with random
        W_bound = np.sqrt(6./(fan_in+fan_out))

        self.W = theano.shared( np.asarray(
                                    rng.uniform(low= -W_bound, high= W_bound, size= filter_shape),
                                    dtype= theano.config.floatX),
                                borrow= True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value= b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(input= input, filters= self.W,
                          filter_shape= filter_shape,
                          input_shape= image_shape)

        # pool each feature map individually, using maxpooling

        pooled_out = pool.pool_2d(input= conv_out, ds= poolsize, ignore_border= True)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
