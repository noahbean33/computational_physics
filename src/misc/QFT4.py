""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code by G. He.), R Landau, Oregon State Unv, MJ Paez, 
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""     
    
# QFT4.py: A Qiskit program to compute a 2-qubit QFT for 4 components

  import math
  from qiskit import QuantumCircuit
  import qiskit.quantum_info as qi
  import numpy as np

  def qft2(inverse=False) -> QuantumCircuit:
      angle = math.pi/2
      if inverse is True:  angle = -angle
      qc = QuantumCircuit(2)             # Create a 2-qubit circuit
      qc.h(1)                            # H gate on qubit-1
      qc.cp(angle, 0, 1)                 # Controlled phase gate
      qc.h(0)
      qc.swap(0, 1)                      # Swap the qubits
      return qc                          # Return circuit as gate
 
  if __name__ == "__main__":
      print(np.around(qi.Operator(qft2()).data, 3)) # Circuit matrix