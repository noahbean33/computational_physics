""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# Hubble.py:  Fit to Hubble dat, adapted from Campesato tensorflow 2 primer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

r = tf.Variable([0.032,0.034,0.214,0.263, 0.275, 0.275, 0.45, 0.5, 0.5,\
	0.63,0.8,0.9,0.9,0.9,0.9, 1.0,1.1,1.1,1.4,1.7,2.0,2.0,2.0,2.0]) # R 
v = tf.Variable([170.,290.,-130.,-70.,-185.,-220.,200.,290.,270.,200.,300.,
     -30.,650.,150.,500.,920.,450.,500.,500.,960. ,500.,850.,800.,1090.]) 
m = tf.Variable(0.)    # Init m, b;   y = mx + b
b = tf.Variable(0. )
slope = 500.
bias = 0.0
step = 10
learning_rate = 0.02
steps = 300
x_train = r
print(x_train)
y_train = slope*x_train + bias

def predict_y_value(x):                                              # y(x) 
    y = m*x + b
    return y
def squared_error(y_pred, y_true):                    # Sum squared errors
    return tf.reduce_mean(tf.square(y_pred - y_true))

loss = squared_error(predict_y_value(x_train), y_train)	
for i in range(steps):
 with tf.GradientTape() as tape:
   predictions = predict_y_value(x_train)
   loss = squared_error(predictions, y_train)
 gradients = tape.gradient(loss, [m, b])
 m.assign_sub(gradients[0] * learning_rate)
 b.assign_sub(gradients[1] * learning_rate)
 if(i % step) == 0:
   print("Step %d, Loss %f, m %f " % (i, loss.numpy(),m))
   y = m*x_train + b
   plt.xlabel("r Mpc")
   plt.ylabel("v km/s")
   plt.scatter(r, v)
   plt.plot(x_train, y)
 plt.show()	 