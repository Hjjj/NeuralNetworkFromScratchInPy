import unittest
import math
from Neuron import Neuron

# FILE: test_Neuron.py


class TestNeuron(unittest.TestCase):
    
    def test_init(self):
        weights = [0.5, -0.6]
        bias = 0.1
        neuron = Neuron(weights, bias)
        self.assertEqual(neuron.weights, weights)
        self.assertEqual(neuron.bias, bias)
        self.assertIsNone(neuron.output)
    
    def test_activate(self):
        weights = [0.5, -0.6]
        bias = 0.1
        neuron = Neuron(weights, bias)
        inputs = [1, 2]
        expected_output = 1 / (1 + math.exp(-(0.5*1 + (-0.6)*2 + 0.1)))
        self.assertAlmostEqual(neuron.activate(inputs), expected_output)
    
    def test_sigmoid(self):
        neuron = Neuron([], 0)
        x = 0
        expected_output = 1 / (1 + math.exp(-x))
        self.assertAlmostEqual(neuron.sigmoid(x), expected_output)

        def test_sigmoid_derivative(self):
            neuron = Neuron([], 0)
            x = 0
            sigmoid_value = neuron.sigmoid(x)
            expected_output = sigmoid_value * (1 - sigmoid_value)
            self.assertAlmostEqual(neuron.sigmoid_derivative(x), expected_output)
        
        def test_relu(self):
            neuron = Neuron([], 0)
            x = 1
            expected_output = max(0, x)
            self.assertEqual(neuron.relu(x), expected_output)
        
        def test_relu_derivative(self):
            neuron = Neuron([], 0)
            x = 1
            expected_output = 1 if x > 0 else 0
            self.assertEqual(neuron.relu_derivative(x), expected_output)
        
        def test_tanh(self):
            neuron = Neuron([], 0)
            x = 0
            expected_output = math.tanh(x)
            self.assertAlmostEqual(neuron.tanh(x), expected_output)
        
        def test_tanh_derivative(self):
            neuron = Neuron([], 0)
            x = 0
            expected_output = 1 - math.tanh(x) ** 2
            self.assertAlmostEqual(neuron.tanh_derivative(x), expected_output)



if __name__ == '__main__':
    unittest.main()