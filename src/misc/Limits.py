""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# Limits.py: determines approximate machine precision

N = 10
eps = 1.0
for i in range(N):
    eps = eps/2
    one_Plus_eps = 1.0  +  eps
    print('eps = ', eps, ', one  +  eps = ', one_Plus_eps)