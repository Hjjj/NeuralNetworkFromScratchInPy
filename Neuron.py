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
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.
        """
        sigmoid_value = self.sigmoid(x)
        return sigmoid_value * (1 - sigmoid_value)
    
    def relu(self, x):
        """
        ReLU activation function.
        """
        return max(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU activation function.
        """
        return 1 if x > 0 else 0
    
    def tanh(self, x):
        """
        Tanh activation function.
        """
        return math.tanh(x)

    def tanh_derivative(self, x):
        """
        Derivative of the tanh activation function.
        """
        return 1 - math.tanh(x) ** 2




