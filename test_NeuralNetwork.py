import unittest
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
        expected_output = ...  # Calculate the expected output based on the weights and biases
        self.assertAlmostEqual(output, expected_output)

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
    
    def test_train(self):
        nn = NeuralNetwork(3, 2, 1)
        nn.set_weights('hidden', 0, [0.2, 0.8, -0.5], 0.1)
        nn.set_weights('hidden', 1, [0.5, -0.91, 0.26], -0.2)
        nn.set_weights('output', 0, [0.1, -0.3], 0.3)
        
        features = [[1.0, 0.5, -1.5], [0.5, -1.0, 1.5]]
        expected_outputs = [[0.5], [0.1]]
        
        nn.train(features, expected_outputs, learning_rate=0.1, epochs=1000)
        
        # Check if the weights and biases have been updated (values will depend on the training process)
        # Replace the expected weights and biases with the actual expected values after training
        expected_hidden_weights_0 = [...]  # Expected weights for hidden neuron 0
        expected_hidden_bias_0 = ...  # Expected bias for hidden neuron 0
        expected_hidden_weights_1 = [...]  # Expected weights for hidden neuron 1
        expected_hidden_bias_1 = ...  # Expected bias for hidden neuron 1
        expected_output_weights_0 = [...]  # Expected weights for output neuron 0
        expected_output_bias_0 = ...  # Expected bias for output neuron 0
        
        self.assertEqual(nn.hidden_neurons[0].weights, expected_hidden_weights_0)
        self.assertEqual(nn.hidden_neurons[0].bias, expected_hidden_bias_0)
        self.assertEqual(nn.hidden_neurons[1].weights, expected_hidden_weights_1)
        self.assertEqual(nn.hidden_neurons[1].bias, expected_hidden_bias_1)
        self.assertEqual(nn.output_neurons[0].weights, expected_output_weights_0)
        self.assertEqual(nn.output_neurons[0].bias, expected_output_bias_0)

if __name__ == '__main__':
    unittest.main()