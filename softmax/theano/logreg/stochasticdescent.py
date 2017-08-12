import numpy as np
import theano, timeit, os
import theano.tensor as T
import six.moves.cPickle as pickle

class stochasticdescent(object):
    """docstring for stochasticdescent"""
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
        self.n_test_batches  = self.test_x.get_value(borrow= True).shape[0]  //self.batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()         # index to minibatch
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.test_model = theano.function(inputs    = [index],
            outputs   = self.classifier.errors(self.y),
            givens    = {self.x: self.test_x[index * self.batch_size: (index + 1)* self.batch_size],
                         self.y: self.test_y[index * self.batch_size: (index + 1)* self.batch_size] }
            )

        self.validate_model = theano.function(inputs    = [index],
            outputs   = self.classifier.errors(self.y),
            givens    = {self.x: self.valid_x [index * self.batch_size: (index + 1)* self.batch_size],
                         self.y: self.valid_y [index * self.batch_size: (index + 1)* self.batch_size] }
            )

        self.grad_W = T.grad(cost= self.cost, wrt= self.classifier.W)
        self.grad_b = T.grad(cost= self.cost, wrt= self.classifier.b)
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.classifier.W, self.classifier.W - self.learning_rate* self.grad_W),
                   (self.classifier.b, self.classifier.b - self.learning_rate* self.grad_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(inputs   = [index],
        outputs  = self.cost,
        updates  = updates,
        givens   = {self.x: self.train_x[index * self.batch_size: (index + 1)* self.batch_size],
                    self.y: self.train_y[index * self.batch_size: (index + 1)* self.batch_size]}
        )
        
    def train(self):
        print("################# Training #################")
        patience = 5000         ## look as this many examples regardless
        patience_increase = 2   ## wait this much longer when a new best is found

        improvement_threshold = 0.995 ## a relative improvement of this much is considered significant
        validation_frequency = min(self.n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
        self.best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < self.n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(self.n_train_batches):
                minibatch_avg_cost = self.train_model(minibatch_index)

                iter = (epoch - 1)* self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [self.validate_model(i) for i in range(self.n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            self.n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    if this_validation_loss < self.best_validation_loss:
                        if this_validation_loss < self.best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        self.best_validation_loss = this_validation_loss

                        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
                        test_score = np.mean(test_losses)

                        print( ('     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                                ) %
                                (   epoch,
                                    minibatch_index + 1,
                                    self.n_train_batches,
                                    test_score * 100.
                                )
                        )

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (self.best_validation_loss * 100., test_score * 100.)
        )
        # save the best model
        with open(self.best_model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.1fs' % ((end_time - start_time))))
