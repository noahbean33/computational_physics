""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# ODEsympy.py: symbolic soltn of HO ODE using sympy

from sympy import *

f, g = symbols('f g',cls=Function)      # makes f a function 
t, kap, w0 = symbols('t kap w0')
f(t)
f(t).diff(t)
diffeq = Eq(f(t).diff(t,t) + kap*(f(t).diff(t),0) + w0*w0*f(t))  # ODE
print ("\n ODE to be solved:") 
print (diffeq)
print ("\n Solution to ODE:")
ff = dsolve(diffeq,f(t))     # Solves ODE
F = ff.subs(t,0)
print (ff)