import numpy as np
import theano
import os
import theano.tensor as T
from load_dataset import *
from logreg.logreg import logreg
from logreg.stochasticdescent import stochasticdescent


n_in= 28*28
n_out= 10
dataset = '../datasets/mnist.pkl.gz'
learning_rate= 0.13
batch_size= 600

def predict(classifier, x):
    classifier = pickle.load(open('best_model.pkl'))
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
    classifier = logreg(input=x, n_in= n_in, n_out= n_out)
    cost = classifier.negative_log_likelihood(y)

    trainer = stochasticdescent(x, y, datasets, classifier, cost, learning_rate, batch_size)
    trainer.train()

    best_model_path = 'best_model.pkl';
    if os.path.isfile(best_model_path):
        classifier = pickle.load(open(best_model_path))
        predict(classifier, trainer.test_x.get_value())
    else:
        print("couldn't find trained model")
