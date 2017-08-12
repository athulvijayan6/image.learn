import theano
import theano.tensor as T
import numpy as np
from mlp.hiddenlayer import hiddenlayer
from logreg.logreg import logreg
import timeit, os
import dill
import six.moves.cPickle as pickle


from PooledConvolutionLayer import PooledConvolutionLayer

class cnn(object):
    """docstring for cnn."""
    def __init__(self, x, y, rng, n_out, nkerns, input_size,
                 filter_size, poolsize, datasets, batch_size=500, learning_rate= 0.13, n_epochs=1000):
        self.x = x
        self.y = y
        self.rng = rng
        self.nkerns = nkerns
        self.input_size = input_size
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.poolsize = poolsize
        self.n_out = n_out
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        self.train_x, self.train_y = datasets[0]
        self.valid_x, self.valid_y = datasets[1]
        self.test_x , self.test_y   = datasets[2]

        self.n_train_batches = self.train_x.get_value(borrow= True).shape[0] // self.batch_size
        self.n_valid_batches = self.valid_x.get_value(borrow= True).shape[0] // self.batch_size
        self.n_test_batches  = self.test_x.get_value(borrow= True).shape[0]  //self.batch_size

    def generate_model(self):
        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = self.x.reshape((self.batch_size, self.input_size[2], self.input_size[0], self.input_size[1]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.convlayers = []
        for i, kerns in enumerate(self.nkerns):
            if i == 0:
                layer0 = PooledConvolutionLayer(layer0_input,
                    rng= self.rng,
                    image_shape= (self.batch_size, self.input_size[2], self.input_size[0], self.input_size[1]),
                    filter_shape= (self.nkerns[0], self.input_size[2], self.filter_size[0], self.filter_size[1]),
                    poolsize= self.poolsize
                    )
                image_shape_new = ((self.input_size[0] - self.filter_size[0] + 1)/self.poolsize[0],
                                   (self.input_size[1] - self.filter_size[1] + 1)/self.poolsize[1])

                self.convlayers.append(layer0)
            else:
                layer = PooledConvolutionLayer(self.convlayers[i -1].output,
                    rng= self.rng,
                    image_shape= (self.batch_size, self.nkerns[i-1], image_shape_new[0], image_shape_new[1]),
                    filter_shape= (self.nkerns[i], self.nkerns[i-1], self.filter_size[0], self.filter_size[1]),
                    poolsize= self.poolsize)
                image_shape_new = ((image_shape_new[0] - self.filter_size[0] + 1)/self.poolsize[0],
                                   (image_shape_new[1] - self.filter_size[1] + 1)/self.poolsize[1])
                self.convlayers.append(layer)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.

        fc_layer_input = self.convlayers[-1].output.flatten(2)
        self.fc_layer = hiddenlayer(self.rng,
                               input= fc_layer_input,
                               n_in= self.nkerns[-1]*np.prod(image_shape_new),
                               n_out= self.batch_size,
                               activation= T.tanh)

        self.layer_out = logreg(input= self.fc_layer.output, n_in= self.batch_size, n_out= self.n_out)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer_out.negative_log_likelihood(self.y)

        # allocate symbolic variables for the data
        index = T.lscalar()         # index to minibatch
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.test_model = theano.function(inputs    = [index],
            outputs   = self.layer_out.errors(self.y),
            givens    = {self.x: self.test_x[index * self.batch_size: (index + 1)* self.batch_size],
                         self.y: self.test_y[index * self.batch_size: (index + 1)* self.batch_size] }
            )

        self.validate_model = theano.function(inputs    = [index],
            outputs   = self.layer_out.errors(self.y),
            givens    = {self.x: self.valid_x [index * self.batch_size: (index + 1)* self.batch_size],
                         self.y: self.valid_y [index * self.batch_size: (index + 1)* self.batch_size] }
            )

        self.params = self.fc_layer.params + self.layer_out.params

        for layer in self.convlayers:
            self.params += layer.params

        self.grads = T.grad(self.cost, self.params)

        self.updates = [ (param_i, param_i - self.learning_rate*grad_i)
                        for param_i, grad_i in zip(self.params, self.grads)]

        self.train_model = theano.function([index],
                                           self.cost,
                                           updates=self.updates,
                                           givens={
                                            self.x: self.train_x[index*self.batch_size: (index+1)*self.batch_size],
                                            self.y: self.train_y[index*self.batch_size: (index+1)*self.batch_size]
                                           }
                            )

    def train(self):
        print("################# Training #################")
        patience = 1000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(self.n_train_batches, patience // 2)

        self.best_validation_loss = np.inf
        self.best_iter = 0
        self.test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while ( epoch < self.n_epochs ) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):
                minibatch_avg_cost = self.train_model(minibatch_index)

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [self.validate_model(i) for i in range(self.n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        ( epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.)
                    )

                    if (this_validation_loss < self.best_validation_loss):
                        if (this_validation_loss < self.best_validation_loss * improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        self.best_validation_loss = this_validation_loss
                        self.best_iter = iter

                        # test it
                        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
                        self.test_score = np.mean(test_losses)
                        print(  ('     epoch %i, minibatch %i/%i, test error of best model %f %%') %
                                (epoch, minibatch_index + 1, self.n_train_batches, self.test_score * 100.)
                             )

                if patience <= iter:
                    done_looping = True
                    break
        end_time = timeit.default_timer()
        print ( ('Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%') %
                (self.best_validation_loss * 100., self.best_iter + 1, self.test_score * 100.)
        )

        print(('The code for file ' + os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.))
             )
