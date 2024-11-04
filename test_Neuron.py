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

if __name__ == '__main__':
    unittest.main()