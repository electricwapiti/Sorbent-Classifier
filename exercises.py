# This is a set of exercises I did to follow the tutorial
import numpy as np
import random
import os
# import sys

# Import the files from the parent directory (..)
from mnist_loader import load_data_wrapper
from Network import Network

# Load the MNIST data
training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 10])

# Train the network
net.SGD(training_data,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test_data)