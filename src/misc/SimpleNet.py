""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# SimpleNet.py:  A simple neuron network

import numpy as np
def f(x):  return 1/(1 + np.exp(-x))       # Sigmoid activation function
def fprime(x):   return np.exp(-x)/(1 + np.exp(-x))**2  # Sigmoid derivs
def Loss(y_true, y_out):
    los = ((y_true - y_out)**2).mean()
    #print(los)
    return los

class SimpleNet:                     # x_1, x_2 in, hidden h1, h2, y_out
   def __init__(self):                                    # Random inits
     self.w1=np.random.normal()                                # Weights
     self.w2=np.random.normal()
     self.w3=np.random.normal()
     self.w4=np.random.normal()
     self.w5=np.random.normal()
     self.w6=np.random.normal()
     self.b1=np.random.normal()                                 # Biases
     self.b2=np.random.normal()
     self.b3=np.random.normal()

   def feedfwd(self, x):
      h1   = f(self.w1*x[0]  + self.w2*x[1] + self.b1)
      h2   = f(self.w3*x[0]  + self.w4*x[1] + self.b2)
      out  = f(self.w5*h1    + self.w6*h2   + self.b3)
      return out

   def train (self, data, all_y_trues):
      learn_rate = 0.1
      N = 1000                                 # Number of learning loops
      for n in range(N):
         for x, y_true in zip(data, all_y_trues):
            sum_h1   = self.w1*x[0] + self.w2*x[1] + self.b1
            h1       = f(sum_h1)
            sum_h2   = self.w3*x[0] + self.w4*x[1] + self.b2
            h2       = f(sum_h2)
            sum_out  = self.w5*h1 + self.w6*h2 + self.b3
            out      = f(sum_out)
            y_out    = out
            d_L_d_yout =-2*(y_true-y_out)                  # Partial deriv
            d_yout_d_w5 = h1*fprime(sum_out)               # Output neuron
            d_yout_d_w6 = h2*fprime(sum_out)
            d_yout_d_b3 = fprime(sum_out)
            d_yout_d_h1 = self.w5*fprime(sum_out)
            d_yout_d_h2 = self.w6*fprime(sum_out )
            d_h1_d_w1 = x[0]*fprime(sum_h1)             # Hidden Neuron h1
            d_h1_d_w2 = x[1]*fprime(sum_h1)
            d_h1_d_b1 = fprime(sum_h1)
            d_h2_d_w3 = x[0]*fprime(sum_h2)             # Hidden Neuron h2
            d_h2_d_w4 = x[1]*fprime(sum_h2)
            d_h2_d_b2 = fprime(sum_h2)
            # Update weights and biases
            self.w1 -= learn_rate*d_L_d_yout*d_yout_d_h1*d_h1_d_w1     # h1
            self.w2 -= learn_rate*d_L_d_yout*d_yout_d_h1*d_h1_d_w2
            self.b1 -= learn_rate*d_L_d_yout*d_yout_d_h1*d_h1_d_b1
            self.w3 -= learn_rate*d_L_d_yout*d_yout_d_h2*d_h2_d_w3     # h2
            self.w4 -= learn_rate*d_L_d_yout*d_yout_d_h2*d_h2_d_w4
            self.b2 -= learn_rate*d_L_d_yout*d_yout_d_h2*d_h2_d_b2
            self.w5 -= learn_rate * d_L_d_yout * d_yout_d_w5      # Out n's
            self.w6 -= learn_rate * d_L_d_yout * d_yout_d_w6
            self.b3 -= learn_rate * d_L_d_yout * d_yout_d_b3
         if (n%10) == 0:                                  # Loss at loop ends
           y_outs = np.apply_along_axis(self.feedfwd,1,data)
           TotLoss =Loss(all_y_trues,y_outs)
           #print("resta",((all_y_trues- y_outs)**2).mean())
           #print("y_trues",all_y_trues)
           print(" Loop n = %d Loss: %.3f" % (n, TotLoss))
data = np.array ([[-2, -1], [25, 6], [17, 4],[-15, -6] ])     # Input Data
all_y_trues = np.array ([ 1, 0, 0, 1 ])
network = SimpleNet()                                           # Train net
network.train(data, all_y_trues)			