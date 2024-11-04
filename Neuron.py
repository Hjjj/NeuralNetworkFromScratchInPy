import math

class Neuron(object):
    def __init__(self, weights, bias):
        self.weights = weights  # List of weights for the neuron's inputs
        self.bias = bias        # Bias term for the neuron
        self.output = None      # Output value of the neuron after activation

    def activate(self, inputs):
        """
        Calculate the neuron's output using the weighted sum of inputs and bias.
        """
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(z)
        return self.output

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + math.exp(-x))




