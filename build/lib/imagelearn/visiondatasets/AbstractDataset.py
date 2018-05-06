import abc


class AbstractDataset(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_batch(self, batch_size, num_epochs, is_training):
        """
        Override this to create graph for serving a batch of data
        :param batch_size:
        :param num_epochs:
        :param is_training:
        :return: batch
        """

    @abc.abstractmethod
    def download_and_convert(self):
        """
        Override this to create function for downloading and converting to tfrecords file
        :return: None
        """