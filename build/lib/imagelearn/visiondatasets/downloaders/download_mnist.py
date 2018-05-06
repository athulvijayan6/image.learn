import logging
import os
import urllib

_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
_out_file_name = "mnist.pkl.gz"


def download_mnist(data_dir):
    file_name = os.path.join(data_dir, _out_file_name)
    if os.path.isfile(file_name):
        logging.info("File already exists. Skipping download")
        return
    urllib.request.urlretrieve(_url, file_name)
    logging.info("Downloading data set complete.")


if __name__ == "__main__":
    _data_dir = "/tmp/mnist"
    if not os.path.isdir(_data_dir):
        os.makedirs(_data_dir)
    download_mnist(_data_dir)
