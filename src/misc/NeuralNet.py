""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# NeuralNet.py:  A simple AI neural network 

import numpy as np

def f(x) : return 1./ (1. + np.exp(-x))     # Activation function
class Neuron :
   def __init__(self, weights, bias) :
       self.weights = weights
       self.bias = bias
   def feedforward(self, inputs) :                 # Process input
      Sum = np.dot (self.weights , inputs) + self.bias
      return f(Sum)

weights = np.array([-1., 1.])                     # w1= -1, w2 = 1
bias = 0
n = Neuron(weights ,bias)
x = np.array([12 ,8])                             # x1 = 12 , x2=8
print(n.feedforward(x))              # Output: 0.01798620996209156

class NeuralNetwork:   # 2-neuron network, 2 hidden layers, 1 output
  def __init__(self):
    weights = np.array([0,1])
    bias = 0
    self.h1 = Neuron(weights, bias)         # Neuron class as before
    self.h2 = Neuron(weights, bias)
    self.O  = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)
    out_out  = self.O.feedforward(np.array([out_h1, out_h2]))
    return out_out
network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))            #output: 0.7216325609518421