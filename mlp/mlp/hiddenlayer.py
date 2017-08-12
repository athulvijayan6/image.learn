# @Author: Athul Vijayan <athul>
# @Date:   2017-01-27T01:07:18+05:30
# @Last modified by:   athul
# @Last modified time: 2017-01-27T02:02:06+05:30



import numpy as np
import theano
import theano.tensor as T
import os, gzip, sys, timeit
import six.moves.cPickle as pickle

class hiddenlayer(object):
    """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.activation = activation
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            W_values = np.asarray(rng.uniform(low     = -np.sqrt(6. / (n_in + n_out)),
                                                high    = np.sqrt(6. / (n_in + n_out)),
                                                size    = (n_in, n_out)
                                                ),
                                  dtype= theano.config.floatX
                                 )
            if activation == T.nnet.sigmoid:
                W_values = W_values * 4

            W = theano.shared(value= W_values, name= 'W', borrow= True)
        if b is None:
            b_values = np.zeros((n_out, ), dtype= theano.config.floatX)
            b = theano.shared(value= b_values, name= 'b', borrow= True)
        self.W = W
        self.b = b

        linear_output = T.dot(self.input, self.W) + self.b

        self.output = (linear_output if activation is None else activation(linear_output))

        self.params = [self.W, self.b]
