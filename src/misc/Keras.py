""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# Keras.py: Linear regression fit to Hubble data with Keras

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras.layers import Dense
import numpy as np
                      # Data
r = [0.032,0.034,0.214,0.263,.275,.275,.45,.5,.5,.63,.8,.9,.9,.9,.9,
   1.0,1.1,1.1,1.4,1.7,2.0,2.0,2.0,2.0]           # Distance Mparse
v = [170.,290.,-130.,-70.,-185.,-220.,200.,290.,270.,200.,300.,
     -30.,650.,150.,500.,920.,450.,500.,500.,960.
     ,500.,850.,800.,1090.]  # Recession velocity km/s
# Create the model: Sequential() only 1 dense layer
layer0 = tf.keras.layers.Dense(units=1,input_shape=[1])
model = tf.keras.Sequential([layer0])
model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(1))	
history = model.fit(r,v,epochs=2000,verbose=0)									
plt.plot(history.history['loss'])
plt.xlabel("Epochs number")
plt.ylabel("Loss")
plt.show()
weights = layer0.get_weights()
weight = weights[0][0]
bias = weights[1]
print('weight: {} bias: {}'.format(weight, bias))
y_learned = r * weight + bias
plt.scatter(r, v, c='blue')
plt.plot(r, y_learned,color='r')
plt.show()
weights = layer0.get_weights()
weight = weights[0][0]
bias = weights[1]
print('weight: {} bias: {}'.format(weight, bias))
y_learned = r * weight + bias
# Output: weight: [448.52048] bias: [-34.726036]
plt.scatter(r, v, c='blue')
plt.plot(r, y_learned,color='r')
plt.show()