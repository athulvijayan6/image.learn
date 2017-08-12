import numpy as np
import theano
import os, dill
import theano.tensor as T
from load_dataset import *
from logreg import logreg
from mlp.backprop import backprop
from mlp.mlp import mlp


n_in= 28*28
n_out= 10
n_hidden= 500
dataset = '../datasets/mnist.pkl.gz'
learning_rate= 0.13
batch_size= 600
L1_reg = 0.00
L2_reg = 0.0001
n_epochs = 1

def predict(classifier, x):
    classifier = dill.load(open('best_model.pkl'))
    # compile a predictor function
    predict_model = theano.function(inputs= [classifier.input],
                                    outputs= classifier.y_pred)
    predicted_values = predict_model(x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

if __name__=="__main__":
    datasets = load_data_mnist(dataset)
    x = T.matrix('x')           # data, presented as rasterized images
    y = T.ivector('y')          # labels, presented as 1D vector of [int] labels
    rng = np.random.RandomState(1234)
    classifier = mlp(rng= rng, input=x, n_in= n_in, n_hidden=n_hidden, n_out= n_out)
    cost = (classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)

    trainer = backprop(x, y, datasets, classifier, cost, learning_rate, batch_size, n_epochs= n_epochs)
    trainer.train()

    best_model_path = 'best_model.pkl';
    if os.path.isfile(best_model_path):
        classifier = pickle.load(open(best_model_path))
        predict(classifier, trainer.test_x.get_value())
    else:
        print("couldn't find trained model")
