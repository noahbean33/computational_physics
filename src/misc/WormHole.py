""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# WormHole.py: Sympy evaluation of derivative for Ellis Wormhole 

from sympy import *
L, x, M, rho, a, r, I ,lp= symbols('L x M rho a r I lp')
x = (2*L-a)/(pi*M)
r = rho+M*(x*atan(x) -log(1+x*x)/2)
p = diff(r,L)
print(p)
print("hola")
n = simplify(p)
print(n)
v = integrate(sqrt(1-n*n),(L,0,lp))
print("integral",v) 
# Result: 2*atan((2*L - a)/(pi*M))/pi