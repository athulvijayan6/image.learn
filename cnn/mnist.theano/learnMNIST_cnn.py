from load_dataset import *
from cnn.cnn import cnn
import dill
import six.moves.cPickle as pickle

dataset = '../datasets/mnist.pkl.gz'

def predict(classifier, x):
    classifier = dill.load(open('best_model.pkl'))
    # compile a predictor function
    predict_model = theano.function(inputs= [classifier.input],
                                    outputs= classifier.y_pred)
    predicted_values = predict_model(x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__=='__main__':
    datasets = load_data_mnist(dataset)
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)
    n_out = 10
    nkerns = np.asarray([1, 1])
    input_size = (28, 28, 1)
    filter_size = (5, 5)
    poolsize = (2, 2)
    n_epochs = 2
    best_model_path='best_model.pkl'

    convnet = cnn(x, y, rng, n_out, nkerns, input_size,
                 filter_size, poolsize, datasets, n_epochs=n_epochs)
    convnet.generate_model()
    convnet.train()
