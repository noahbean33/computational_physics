""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# Entangle.py: Calculate entangled quantum states

from numpy import*;  from numpy.linalg import*

nmax = 4
H = zeros((nmax,nmax), float)
XAXB = array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])      # sigxA.sigxB
YAYB = array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])    # sigyA.sigyA
ZAZB = array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])    # sigzA.sigzA
SASB = XAXB + YAYB + ZAZB -3*ZAZB                     # Hamiltonian/factor
print('Hamiltonian without mu^2/r^3 factor \n',SASB)
es,ev = eig(SASB)                                  # Eigenvalues & vectors
print('Eigenvalues \n',es)
print('Eigenvectors(incolumns)\n',ev)
phi1 = (ev[0,0], ev[1,0], ev[2,0], ev[3,0])              # Extract vectors
phi4 = (ev[0,1], ev[1,1], ev[2,1], ev[3,1])
phi3 = (ev[0,2], ev[1,2], ev[2,2], ev[3,2])
phi2 = (ev[0,3], ev[1,3], ev[2,3], ev[3,3])
basis = [phi1,phi2,phi3,phi4]                          # List eigenvectors
for i in range ( 0 , nmax ) :                   # Hamiltonian in new basis
   for j in range ( 0 ,nmax ) :
      term = dot (SASB, basis [i ] )
      H[i,j] = dot (basis [j],term )
print('Hamiltonian in Eigenvector Basis \n' , H)