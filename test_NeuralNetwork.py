import unittest
import math
from NeuralNetwork import NeuralNetwork
from Neuron import Neuron

class TestNeuralNetwork(unittest.TestCase):

    def test_init(self):
        input_size = 3
        hidden_size = 2
        output_size = 1
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        
        self.assertEqual(len(nn.input_neurons), input_size)
        self.assertEqual(len(nn.hidden_neurons), hidden_size)
        self.assertEqual(len(nn.output_neurons), output_size)

    def test_set_weights_input_layer(self):
        nn = NeuralNetwork(3, 2, 1)
        weights = [0.1, 0.2, 0.3]
        bias = 0.4
        nn.set_weights('input', 0, weights, bias)
        
        self.assertEqual(nn.input_neurons[0].weights, weights)
        self.assertEqual(nn.input_neurons[0].bias, bias)

    def test_set_weights_hidden_layer(self):
        nn = NeuralNetwork(3, 2, 1)
        weights = [0.1, 0.2, 0.3]
        bias = 0.4
        nn.set_weights('hidden', 0, weights, bias)
        
        self.assertEqual(nn.hidden_neurons[0].weights, weights)
        self.assertEqual(nn.hidden_neurons[0].bias, bias)

    def test_set_weights_output_layer(self):
        nn = NeuralNetwork(3, 2, 1)
        weights = [0.1, 0.2]
        bias = 0.3
        nn.set_weights('output', 0, weights, bias)
        
        self.assertEqual(nn.output_neurons[0].weights, weights)
        self.assertEqual(nn.output_neurons[0].bias, bias)

    def test_forward(self):
        nn = NeuralNetwork(3, 2, 1)
        nn.set_weights('input', 0, [0.1, 0.2, 0.3], 0.4)
        nn.set_weights('hidden', 0, [0.1, 0.2, 0.3], 0.4)
        nn.set_weights('output', 0, [0.1, 0.2], 0.3)
        
        inputs = [1, 2, 3]
        # Assuming the forward method is implemented and returns the output
        output = nn.forward(inputs)
        
        # Replace the expected_output with the actual expected value
        expected_output = [[0.8581489350995123, 0.5], [0.6191200338232346]]
        self.assertAlmostEqual(output[0][0], expected_output[0][0])
        self.assertAlmostEqual(output[0][1], expected_output[0][1])
        self.assertAlmostEqual(output[1][0], expected_output[1][0])

    def test_forward(self):
        nn = NeuralNetwork(3, 2, 1)
        nn.set_weights('hidden', 0, [0.2, 0.8, -0.5], 0.1)
        nn.set_weights('hidden', 1, [0.5, -0.91, 0.26], -0.2)
        nn.set_weights('output', 0, [0.1, -0.3], 0.3)
        
        inputs = [1.0, 0.5, -1.5]
        hidden_outputs, final_outputs = nn.forward(inputs)
        
        # Replace the expected_hidden_outputs and expected_final_outputs with the actual expected values
        expected_hidden_outputs = [...]  # Calculate the expected hidden layer outputs
        expected_final_outputs = [...]  # Calculate the expected final outputs
        
        self.assertEqual(hidden_outputs, expected_hidden_outputs)
        self.assertEqual(final_outputs, expected_final_outputs)

    def test_predict(self):
        nn = NeuralNetwork(3, 2, 1)
        nn.set_weights('hidden', 0, [0.2, 0.8, -0.5], 0.1)
        nn.set_weights('hidden', 1, [0.5, -0.91, 0.26], -0.2)
        nn.set_weights('output', 0, [0.1, -0.3], 0.3)
        
        inputs = [1.0, 0.5, -1.5]
        output = nn.predict(inputs)
        
        # Replace the expected_output with the actual expected value
        expected_output = [...]  # Calculate the expected output based on the weights and biases
        self.assertEqual(output, expected_output)
    

    def test_realistic_model(self):
        # Define a simple neural network for weather prediction
        # Input neurons: temperature, humidity
        # Hidden neurons: 2 neurons
        # Output neuron: chance of rain (0 to 1)

        # Initialize neurons to small random values near zero
        hidden_neurons = [
            Neuron([0.071, 0.14], 0.11),
            Neuron([0.13, 0.12], 0.14),
            Neuron([0.07, -0.10], 0.1),
            Neuron([0.11, 0.09], 0.1),
            Neuron([-0.09, 0.13], 0.12)
        ]

        output_neuron = Neuron([0.08,-0.09,0.11,0.13,0.10], 0.09)

        # Training data: [temperature, humidity] -> chance of rain
        training_data = [
            ([30, 70], 0.8),
            ([25, 60], 0.6),
            ([20, 50], 0.4),
            ([15, 40], 0.2)
        ]

        # Simple training loop (using gradient descent)
        learning_rate = 0.01
        for epoch in range(1000):  # Number of training epochs
            for inputs, expected_output in training_data:
                # Forward pass
                hidden_outputs = [neuron.activate(inputs) for neuron in hidden_neurons]
                output = output_neuron.activate(hidden_outputs)

                # Compute error (mean squared error)
                error = expected_output - output

                # Backward pass (gradient descent)
                # Update output neuron weights and bias
                for i in range(len(output_neuron.weights)):
                    output_neuron.weights[i] += learning_rate * error * hidden_outputs[i]
                output_neuron.bias += learning_rate * error

                # Update hidden neurons weights and biases
                for i, neuron in enumerate(hidden_neurons):
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] += learning_rate * error * output_neuron.weights[i] * hidden_outputs[i] * (1 - hidden_outputs[i]) * inputs[j]
                    neuron.bias += learning_rate * error * output_neuron.weights[i] * hidden_outputs[i] 

        # Inference
            test_inputs = [
            [28, 65],
            [28, 99],
            [28, 5]
        ]
        for test_input in test_inputs:
            hidden_outputs = [neuron.activate(test_input) for neuron in hidden_neurons]
            output = output_neuron.activate(hidden_outputs)
            print(f"Chance of rain for input {test_input}: {output}")

if __name__ == '__main__':
    unittest.main()