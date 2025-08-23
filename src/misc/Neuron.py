""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# Neuron.py:  An AI neuron

import numpy as np

def f(x) : return 1./ (1. + np.exp(-x))    # Activation function
class Neuron :
   def __init__(self, weights, bias) :
       self.weights = weights
       self.bias = bias
   
	def feedforward(self, inputs) :               # Process input
      Sum = np.dot (self.weights , inputs) + self.bias
      return f(Sum)

weights = np.array([-1., 1.])                  # w1 = -1, w2 = 1
bias = 0
n = Neuron(weights ,bias)
x = np.array([12 ,8])                         # x1 = 12 , x2 = 8
print(n.feedforward(x))
# output: 0.01798620996209156