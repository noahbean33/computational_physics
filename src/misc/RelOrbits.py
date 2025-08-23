""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# RelOrbits.py Reltv orbits in a gravitational potential, needs rk4

import matplotlib.pyplot as plt
import numpy as np

dh = 0.03
dt = dh
ell = 4.3 # is el/M
G = 1.0
N = 2
E = -0.028
phi = np.zeros((7000),float)
rr = np.zeros((7000),float)
y = np.zeros((2),float)
y[0] = 0.0692
y[1] = np.sqrt(2*E/ell**2 + 2*G*y[0]/ell**2-G*y[0]**2+2*G*y[0]**3)

def f(t,y):
    rhs = np.zeros(2)
    rhs[0] = y[1]
    rhs[1] = -y[0]+G/ell**2 +3*G*y[0]**2
    return rhs
    
f(0,y)
i = 0 
for fi in np.arange(0,12.0*np.pi,dt):
    y = rk4(fi,dt,N,y,f) 
    rr[i] = (1/y[0])*np.sin(fi)         # Note u = 1/r
    phi[i] = (1/y[0])*np.cos(fi)
    i = i+1
f1 = plt.figure()    
plt.axes().set_aspect('equal')        # Equal aspect ratio 
plt.plot(phi[:900],rr[:900])
plt.show()