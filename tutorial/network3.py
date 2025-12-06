"""pytorch_network3.py
~~~~~~~~~~~~~~~~~~~~~~

A PyTorch-based program for training and running simple neural
networks, converted from the Theano-based network3.py by Michael Nielsen.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return F.relu(z)
# sigmoid and tanh are available in torch.nn.functional


#### Constants
# PyTorch typically runs on the GPU automatically if available.
# We set a device explicitly here for clarity.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

#### Load the MNIST data
def load_data_shared(filename="Tutorial\data\mnist.pkl.gz"):
    """
    Loads MNIST data and converts it into PyTorch DataLoader objects.
    """
    with gzip.open(filename, 'rb') as f:
        # Theano version uses iso-8859-1 encoding, which is retained here.
        training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')

    def shared(data):
        """Converts (input, target) NumPy arrays to PyTorch Tensors."""
        x = torch.from_numpy(data[0]).float()
        y = torch.from_numpy(data[1]).long() # Targets are typically long integers
        return x.to(DEVICE), y.to(DEVICE)

    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(nn.Module):

    def __init__(self, layers, mini_batch_size):
        """
        Takes a list of `layers`, describing the network architecture.
        In PyTorch, we use nn.ModuleList to hold the layers.
        """
        super(Network, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.mini_batch_size = mini_batch_size

        # PyTorch automatically tracks parameters, so 'self.params' is redundant
        # but kept for API consistency.
        self.params = list(self.parameters())

    def forward(self, x, training=False):
        """
        Defines the forward pass through the network.
        Dropout is handled within the layer definition in PyTorch.
        """
        # The original Theano code had separate 'output' and 'output_dropout'
        # variables. In PyTorch, we handle dropout directly inside the forward
        # pass of the layer modules, controlled by model.train() or model.eval().
        for layer in self.layers:
            x = layer(x)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""

        # --- 1. Data Setup for PyTorch DataLoader ---
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # Create DataLoader objects for efficient batching and iteration
        train_loader = DataLoader(TensorDataset(training_x, training_y),
                                  batch_size=mini_batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(validation_x, validation_y),
                                  batch_size=mini_batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y),
                                 batch_size=mini_batch_size, shuffle=False)

        num_training_batches = len(train_loader)

        # --- 2. Define Loss, Optimizer, and Regularization ---
        # The original used Log-Likelihood/Cross-Entropy for Softmax
        criterion = nn.NLLLoss(reduction='mean') # Negative Log Likelihood Loss
        # PyTorch handles L2 (weight decay) regularization directly in the optimizer
        optimizer = optim.SGD(self.parameters(), lr=eta, weight_decay=lmbda)

        # --- 3. Training Loop ---
        best_validation_accuracy = 0.0
        best_iteration = 0
        
        for epoch in range(epochs):
            self.train() # Set network to training mode (enables dropout)
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                iteration = num_training_batches * epoch + batch_idx

                if iteration % 1000 == 0:
                    print(f"Training mini-batch number {iteration}")

                optimizer.zero_grad() # Zero the gradients
                output = self(data)
                
                # Compute loss (PyTorch takes output and target directly)
                # Note: PyTorch NLLLoss expects log probabilities, so the final layer 
                # should output log_softmax, not softmax. The layer definitions below 
                # will need to reflect this.
                loss = criterion(output, targets)
                
                loss.backward() # Compute gradients
                optimizer.step() # Update parameters

            # --- 4. Evaluation Loop (After each epoch) ---
            self.eval() # Set network to evaluation mode (disables dropout)
            
            # Validation Accuracy
            validation_accuracy = self.evaluate(valid_loader)
            
            print(f"Epoch {epoch}: validation accuracy {validation_accuracy:.2%}")

            if validation_accuracy >= best_validation_accuracy:
                print("This is the best validation accuracy to date.")
                best_validation_accuracy = validation_accuracy
                best_iteration = iteration
                
                # Test Accuracy
                if test_data:
                    test_accuracy = self.evaluate(test_loader)
                    print(f'The corresponding test accuracy is {test_accuracy:.2%}')
                    
        print("\nFinished training network.")
        print(f"Best validation accuracy of {best_validation_accuracy:.2%} obtained at iteration {best_iteration}")
        print(f"Corresponding test accuracy of {test_accuracy:.2%}")


    def evaluate(self, data_loader):
        """Computes the accuracy for a given DataLoader."""
        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for data, targets in data_loader:
                output = self(data)
                # Get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(targets.view_as(pred)).sum().item()
                total += targets.size(0)
        
        return correct / total

#### Define layer types

class ConvPoolLayer(nn.Module):
    """
    A PyTorch Conv2D layer followed by a MaxPool2D layer.
    """
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=F.sigmoid):
        """
        filter_shape: (out_channels, in_channels, filter_h, filter_w)
        image_shape: (mini_batch_size, in_channels, image_h, image_w)
        """
        super(ConvPoolLayer, self).__init__()
        
        out_channels = filter_shape[0]
        in_channels = filter_shape[1]
        kernel_size = filter_shape[2:]
        self.activation_fn = activation_fn
        self.poolsize = poolsize
        
        # PyTorch Conv2d: (in_channels, out_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=poolsize)
        
        # Original initialization (Xavier/Kaiming is usually better in PyTorch)
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        stdv = np.sqrt(1.0/n_out)
        self.conv.weight.data.normal_(0, stdv)
        self.conv.bias.data.normal_(0, 1.0) 
        
        self.params = list(self.conv.parameters()) # Retain for API consistency

    def forward(self, x):
        # x is assumed to be (batch_size, channels, height, width)
        # We need to reshape the flat MNIST input (784) before this layer 
        # if it's the first layer, which is handled in the example run.

        # Convolution
        conv_out = self.conv(x)
        
        # Max Pooling
        pooled_out = self.pool(conv_out)
        
        # Activation
        # PyTorch automatically adds the bias during the conv step.
        return self.activation_fn(pooled_out)
        
        # Note: Dropout is typically not applied here, matching the original.

class FullyConnectedLayer(nn.Module):

    def __init__(self, n_in, n_out, activation_fn=F.sigmoid, p_dropout=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Original initialization (similar to PyTorch default for n_out):
        stdv = np.sqrt(1.0/n_out)
        self.linear.weight.data.normal_(0, stdv)
        self.linear.bias.data.normal_(0, 1.0)
        
        self.params = list(self.linear.parameters())

    def forward(self, x):
        # Flatten input if coming from ConvPool layer (handled in Network definition)
        x = x.view(x.size(0), -1) 
        
        # Dropout is applied *to the input* in the original code, but 
        # PyTorch applies it *to the output* or *internally*. 
        # We model the dropout mask application on the input as closely as possible
        # by using nn.Dropout, which only takes effect during self.train().
        
        # Dropout input is only used for the T.dot(..., w) calculation, 
        # which is equivalent to applying dropout before the linear layer.
        
        if self.training:
            # Apply dropout to the input tensor
            x = self.dropout(x)
            # The original code scaled the output by (1-p_dropout) in the non-dropout
            # path, but PyTorch handles this scaling automatically during training.
        
        return self.activation_fn(self.linear(x))

    # PyTorch's loss function handles accuracy implicitly. We do not need the accuracy method here.

class SoftmaxLayer(nn.Module):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        super(SoftmaxLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=p_dropout)

        # Original initialization (zeros)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        
        self.params = list(self.linear.parameters())
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        if self.training:
            x = self.dropout(x)
            
        # For nn.NLLLoss (Negative Log Likelihood Loss), the final layer 
        # must output LOG-SOFTMAX probabilities.
        return F.log_softmax(self.linear(x), dim=1)

    # The cost and accuracy methods are handled by the Network's SGD method 
    # using PyTorch's loss functions and the evaluate helper.


#### Run example (Equivalent to original network3.py example)

if __name__ == '__main__':
    # Fix the data path if needed (this assumes data/mnist.pkl.gz is outside the script's directory)
    training_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10
    
    # Define a utility function to simplify the creation of the full network
    def run_fcl_softmax_example():
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)
        ], mini_batch_size)

        # Move network to the specified device
        net.to(DEVICE)
        
        # Run training
        net.SGD(training_data, 60, mini_batch_size, 0.1, 
                validation_data, test_data)

    print("\n--- Running Fully Connected + Softmax Example ---")
    run_fcl_softmax_example()
    
    
    # Example for a full Convolutional Network (as described in the tutorial)
    def run_conv_example():
        # Re-load data, but reshape the 784-dimensional input into a 1-channel, 28x28 image 
        # for the first ConvPoolLayer.
        training_x, training_y = training_data
        
        # Reshape the 784 vector to (batch_size, 1, 28, 28) for the ConvPool layer
        # Note: PyTorch uses (Batch, Channels, Height, Width)
        input_shape = (-1, 1, 28, 28) 
        
        # The list of layers defines the architecture:
        net = Network([
            # Layer 1: ConvPool
            # filter_shape: (out_ch=20, in_ch=1, filter_h=5, filter_w=5)
            # image_shape: (mb_size, in_ch=1, img_h=28, img_w=28)
            ConvPoolLayer(filter_shape=(20, 1, 5, 5), image_shape=(mini_batch_size, 1, 28, 28), activation_fn=F.relu),
            
            # Layer 2: ConvPool (output from L1 is 20 maps, size 12x12 after pooling)
            # image_shape for L2: (mb_size, in_ch=20, img_h=12, img_w=12)
            ConvPoolLayer(filter_shape=(40, 20, 5, 5), image_shape=(mini_batch_size, 20, 12, 12), activation_fn=F.relu),
            
            # Layer 3: Fully Connected
            # The output of L2 is 40 maps, size 4x4. Total FC input: 40 * 4 * 4 = 640
            FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100, activation_fn=F.relu),
            
            # Layer 4: Softmax Output
            SoftmaxLayer(n_in=100, n_out=10)
        ], mini_batch_size)
        
        net.to(DEVICE)
        
        # The MNIST data must be reshaped before feeding into the network:
        # We need a custom DataLoader that does this reshape (or modify the network's forward pass)
        
        # For demonstration, we simply reshape the training data input tensor
        training_x_conv = training_x.view(training_x.size(0), 1, 28, 28)
        validation_x_conv = validation_x.view(validation_x.size(0), 1, 28, 28)
        test_x_conv = test_x.view(test_x.size(0), 1, 28, 28)
        
        training_data_conv = (training_x_conv, training_y)
        validation_data_conv = (validation_x_conv, validation_y)
        test_data_conv = (test_x_conv, test_y)

        print("\n--- Running Convolutional Network Example ---")
        net.SGD(training_data_conv, 60, mini_batch_size, 0.1, 
                validation_data_conv, test_data_conv)

    # To run the ConvNet example, uncomment the two lines below:
    # print("\n--- Running Convolutional Network Example ---")
    # run_conv_example()