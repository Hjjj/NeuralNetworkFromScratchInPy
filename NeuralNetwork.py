import math
from Neuron import Neuron

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_neurons = [Neuron([0] * input_size, 0) for _ in range(input_size)]
        self.hidden_neurons = [Neuron([0] * input_size, 0) for _ in range(hidden_size)]
        self.output_neurons = [Neuron([0] * hidden_size, 0) for _ in range(output_size)]

    def set_weights(self, layer, neuron_index, weights, bias):
        """
        Set the weights and bias for a specific neuron in a specific layer.
        """
        if layer == 'input':
            self.input_neurons[neuron_index].weights = weights
            self.input_neurons[neuron_index].bias = bias
        elif layer == 'hidden':
            self.hidden_neurons[neuron_index].weights = weights
            self.hidden_neurons[neuron_index].bias = bias
        elif layer == 'output':
            self.output_neurons[neuron_index].weights = weights
            self.output_neurons[neuron_index].bias = bias

    def forward(self, inputs):
        """
        Perform forward propagation through the network.
        """
        # Calculate hidden layer outputs
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_neurons]
        
        # Calculate output layer outputs
        output_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_neurons]
        
        return hidden_outputs, output_outputs

    def predict(self, inputs):
        """
        Predict the output for given inputs.
        """
        _, final_outputs = self.forward(inputs)
        return final_outputs

    def train(self, features, expected_outputs, learning_rate, epochs):
        """
        Train the neural network using the given features and expected outputs.
        """
        for epoch in range(epochs):
            for inputs, expected in zip(features, expected_outputs):
                # Forward propagation
                hidden_outputs, final_outputs = self.forward(inputs)

                # Calculate output layer error
                output_errors = [expected[i] - final_outputs[i] for i in range(len(expected))]

                # Backpropagate the error to the hidden layer
                hidden_errors = [0] * len(self.hidden_neurons)


                for i in range(len(self.hidden_neurons)):
                    error = sum(output_errors[j] * self.output_neurons[j].weights[i] for j in range(len(self.output_neurons)))
                    hidden_errors[i] = error * self.hidden_neurons[i].sigmoid_derivative(hidden_outputs[i])

                # Update weights and biases for output neurons
                for i in range(len(self.output_neurons)):
                    for j in range(len(self.output_neurons[i].weights)):
                        self.output_neurons[i].weights[j] += learning_rate * output_errors[i] * hidden_outputs[j]
                    self.output_neurons[i].bias += learning_rate * output_errors[i]

                # Update weights and biases for hidden neurons
                for i in range(len(self.hidden_neurons)):
                    for j in range(len(self.hidden_neurons[i].weights)):
                        self.hidden_neurons[i].weights[j] += learning_rate * hidden_errors[i] * inputs[j]
                    self.hidden_neurons[i].bias += learning_rate * hidden_errors[i]