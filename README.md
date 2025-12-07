# Introduction
I followed Michael Nielsen's book Neural Networks and Deep Learning. I found a modern python3 version of his tutorial to use as my starter code. The python files Network.py and network2.py implement a neural network from scratch in Python with backpropagation also done from scratch.

# Contents
## generate.py 
generates fake sorbents and scores them according to a formula which the network will learn. It makes a file of these fake sorbents and scores and saves it to a pkl file.
## network.py
is my neural network adapted from the tutorial. It uses Stochastic Gradient Descent as the learning algorithm. It saves the trained network to the json file sorbent_net.json. It uses a method to load the sorbent data imported from load_sorbent_data in sorbent_loader.py
## train.py
is my file to train the neural network. It asks for the parameters for network architecture, epochs, mini batch size & eta.
## test.py
is my file to test the neural network. It loads the trained network and tests it on the test data. Outputs: MSE & R^2.
## predict.py
is my file to ask the user to input a "new sorbent" and to return a score calculated with my neural network.