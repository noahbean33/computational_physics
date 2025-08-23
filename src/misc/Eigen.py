""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""  

# Eigen.py  Solution of matrix eigenvalue problem

from numpy import*
from numpy.linalg import eig

A = array( [[2./3,-1./4,-1./4], [-1./4,2./3,-1./4], [-1./4,-1./4,2./3]] )
print('I =\n', A)

Es, evectors = eig(A)                         # Solves eigenvalue problem
print('Eigenvalues =  ', Es, '\n Matrix of Eigenvectors =\n', evectors) 
Vec = array([ evectors[0, 0], evectors[1, 0], evectors[2, 0] ] )
print('A single eigenvector  =', Vec)
LHS = dot(A, Vec)
RHS = Es[0]*Vec 
print('LHS - RHS =', LHS-RHS) 
