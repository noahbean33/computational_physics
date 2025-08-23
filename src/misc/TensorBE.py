""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# TensorBE.py:  TensorFlow calc H isotope binding E's

import tensorflow as tf
import matplotlib.pyplot as mpl
import numpy as np
B = np.zeros(7)#[0,0,0,0,0,0,0]
mP = tf.constant(938.2592)                                   # Proton mass  
mN = tf.constant(939.5527)                                  # Neutron mass  
mH = tf.multiply(1.00784, 931.494028)              # H atom mass in MeV/c2                                           
am = tf.constant([1.007825032, 2.01401778,\
                  3.016049278, 4.026, 5.035, 6.045, 7.05])    # Iso Masses
A = tf.constant([1, 2., 3., 4., 5., 6., 7.])              # Atomic numbers
for i in range(7):
    C = mH + (i)*mN - am[i]*931.494028
    AN = A[i]
    B[i]= C/AN
    print("BN :",B[i] )
mpl.ylabel('Binding energy per nucleon (MeV)')
mpl.xlabel('Atomic mass number')
mpl.plot(A,B)
mpl.show()