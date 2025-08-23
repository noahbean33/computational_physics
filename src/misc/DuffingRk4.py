""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# rk4Duffing.py solve ODE for Duffing Osc via rk4 & Matplotlib

import numpy as np, matplotlib.pylab as plt  
from math import *

tt=np.zeros((2000),float);  yy=np.zeros((2000),float);     # Declare arrays
vy=np.zeros((2000),float);             
a=0.5                                 
b=-0.5                                
g=0.02                                
F=0.0008                              
w=1.                                  
def f(t,y):                           
    rhs=[0]*(2)                       
    rhs[0]=y[1]
    rhs[1]=-2*g*y[1]-a*y[0]-b*y[0]**3+F*cos(w*t)
    return rhs
    

def rk4(t,h,y,n):       #Runge-Kutta 4, parameters: t, step,y,dimension
    k1=[0.]*(n+1)
    k2=[0.]*(n+1)
    k3=[0.]*(n+1)
    k4=[0.]*(n+1)
    fR=[0.]*(n+1)
    ydumb=[0.]*(n+1)
    fR=f(t, y)                  # function returns rhs
    for i in range(0,n+1):
       k1[i] = h*fR[i]          # Compute function values  assign to k1          
    for i in range(0, n+1):
        ydumb[i] = y[i] + k1[i]/2. #call function for xi+h/2,ydumb
    fR=f(t + h/2.,ydumb)
    for i in range(0, n+1):
        k2[i]=h*fR[i]
        ydumb[i] = y[i] + k2[i]/2.    
    fR=f(t + h/2., ydumb)
    for i in range(0, n+1):
        k3[i]=h*fR[i]
        ydumb[i] = y[i] + k3[i] 
    fR=f(t + h, ydumb)###
    for i in range(0, n+1):
        k4[i]=h*fR[i]
    for i in range(0, n+1):
        y[i] = y[i] + (k1[i] + 2.*(k2[i] + k3[i]) + k4[i])/6. #runge kutta
    return y       # returns y[0]:position and y[1]: velocitu


y=[0]*(2)     # array for 2 values initial conditions
y[0]=0.09     # initial position
y[1]= 0       # initial velocity

f(0.0,y)      # call function for xi=0 with init conds.
dt=0.1        # small time
i=0           # for array 
for t in np.arange(0,200,dt): # this makes 2000 times    
    tt[i]=t
    y=rk4(t,dt,y,1)    # call runge kutta   
    yy[i]= y[0]        #  yy contains positions
    vy[i]=y[1]         # contains velocities
    i=i+1
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,5) )  # bigger figure 

axes[0].plot(tt[1000:],yy[1000:])              # start at 1000: no transients
axes[0].grid()                                 # position vs time
axes[0].set_title('Duffing Oscillator')        # left figure
axes[0].set_xlabel('t')
axes[0].set_ylabel('x(t)')
axes[1].plot(yy[1000:],vy[1000:])           # right figure phase diagram
axes[1].grid()
axes[1].set_title('Phase Diagram Duffing Oscillator')  
axes[1].set_xlabel('x(t)')
axes[1].set_ylabel('v(t)')      
plt.show()
