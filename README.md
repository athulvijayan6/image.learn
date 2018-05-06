<p align="center"><img width=60% src="https://github.com/athulvijayan6/imagelearn/blob/master/logo.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v2.7%20%2F%20v3.6-blue.svg)
![Build Status](https://img.shields.io/travis/USER/REPO.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Basic Overview

A tidy repository of machine learning algorithms for machine vision.
<p align="center"><img width=95% src="https://github.com/anfederico/Waldo/blob/master/media/Schematic.png"></p>

<br>

## Install
```python
pip install <imagelearn-wheel-file.whl>
```

## Algorithms and models
1. CNN - Convolutional neural networks are neural netowk architectures inspired from visual cortex of animal brain. The receptive field of neurons translates to convolution operation.

## How to use
Here is an example of training a basic CNN with MNIST data. You can easily change the model and data set.
```python
from imagelearn.cnnvision.models.CarbonModel import CarbonModel
from imagelearn.visiondatasets.MNIST import MNIST

with tf.Session() as session:
    data_set = MNIST(session=session, data_dir=data_dir)
    if not data_set.has_tfrecords():
        if not data_set.has_input_data():
            data_set.download()
        data_set.write_tfrecords()
    c = CarbonModel(session=session, data_set=data_set, train_dir=train_dir)
    c.build_graph(batch_size=batch_size, num_epochs=num_epochs)
    c.train()
```

## Summary of results
TBD

## Documentation
TBD

## Contributing
Please email athulvijayan6@gmail.com if interested in contributing.
