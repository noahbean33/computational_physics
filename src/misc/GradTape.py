""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# GradTape.py:  Use of Use of Tensor Flow's GradientTape
import tensorflow as tf

m = tf.Variable(1.5)
b = tf.Variable(2.2)
x = tf.Variable(0.5)
y = tf.Variable(1.8)
with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(m, x), b)                          # z = mx +b
    loss = tf.reduce_sum(tf.square(y - z))                # (y-(mx+b))**2
dloss_dx = tape.gradient(loss, x)                              # Gradient
tf.print('dL/dx:', dloss_dx)                  # Output: dL/dx: 3.45000029
tf.print(2*(-m)*(y-(m*x+b)))                          # Output 3.45000029