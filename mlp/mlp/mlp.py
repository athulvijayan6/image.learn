import numpy as np
import theano
import theano.tensor as T
import os, gzip, sys, timeit
import six.moves.cPickle as pickle
from hiddenlayer import hiddenlayer
from logreg import logreg
class mlp(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie


        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.hiddenlayer = hiddenlayer(rng    = rng,
                                       input    = input,
                                       n_in     = self.n_in,
                                       n_out    = self.n_hidden,
                                       activation = T.tanh
                                       )
        self.logreglayer = logreg(input= self.hiddenlayer.output, n_in=n_hidden, n_out= n_out)

        self.L1 = (abs(self.hiddenlayer.W).sum() + abs(self.logreglayer.W).sum())
        self.L2_sqr = ((self.hiddenlayer.W ** 2).sum() + (self.logreglayer.W ** 2).sum())
        self.negative_log_likelihood = (self.logreglayer.negative_log_likelihood)
        self.errors = (self.logreglayer.errors)
        self.params = self.hiddenlayer.params + self.logreglayer.params
        self.y_pred = self.logreglayer.y_pred

        self.input = input
