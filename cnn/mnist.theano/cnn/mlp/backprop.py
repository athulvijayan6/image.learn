import theano
import theano.tensor as T
import numpy as np
import timeit, os
import dill

class backprop(object):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    def __init__(self, x, y, datasets, classifier, cost, learning_rate, batch_size,
                 best_model_path='best_model.pkl', n_epochs= 1000):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cost = cost
        self.classifier = classifier
        self.best_model_path = best_model_path

        self.train_x, self.train_y = datasets[0]
        self.valid_x, self.valid_y = datasets[1]
        self.test_x , self.test_y   = datasets[2]

        self.n_train_batches = self.train_x.get_value(borrow= True).shape[0] // self.batch_size
        self.n_valid_batches = self.valid_x.get_value(borrow= True).shape[0] // self.batch_size
        self.n_test_batches  = self.test_x.get_value(borrow= True).shape[0]  // self.batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()         # index to minibatch
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch

        self.test_model = theano.function(inputs    = [index],
            outputs   = self.classifier.errors(self.y),
            givens    = {self.x: self.test_x[index * self.batch_size: (index + 1)* self.batch_size],
                         self.y: self.test_y[index * self.batch_size: (index + 1)* self.batch_size]}
        )

        self.validate_model = theano.function(inputs= [index],
            outputs= self.classifier.errors(self.y),
            givens= {self.x: self.valid_x[index * self.batch_size: (index + 1)* self.batch_size],
                     self.y: self.valid_y[index * self.batch_size: (index + 1)* self.batch_size]}
        )

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

        self.updates = [(param, param - self.learning_rate * gparam) for
                        param, gparam in zip(self.classifier.params, self.gparams)]

        self.train_model = theano.function(inputs       = [index],
                                           outputs      = self.cost,
                                           updates      = self.updates,
                                           givens       = {self.x: self.train_x[index * self.batch_size: (index + 1)* self.batch_size],
                                                           self.y: self.train_y[index * self.batch_size: (index + 1)* self.batch_size]}
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
        # save the best model
        with open(self.best_model_path, 'wb') as f:
            dill.dump(self.classifier, f, protocol=2)

        print(('The code for file ' + os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.))
             )
