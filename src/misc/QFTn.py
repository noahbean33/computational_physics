""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright G. He, R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# QFT.py: A Qiskit program using n-qubits for 2^n component QFT

import math
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.circuit.library import QFT
import numpy as np

def qft(n: int, inverse: bool = False, skip_swap: bool = False) -> QuantumCircuit:
    # build a n-qubit qft circuit
    angle = np.pi/2
    if inverse is True:
        angle = -angle
    qc = QuantumCircuit(n)
    for i in reversed(range(n)):
        qc.h(i)
        for j in range(i):
            qc.cp(angle/2**(i-j-1), j, i)
    if skip_swap is False:
        for i in range(math.floor(n/2)):
            qc.swap(i, n-i-1)
    return qc

if __name__ == "__main__":
    for k in range(10):
        print(np.max(np.abs(qi.Operator(qft(k)).data-qi.Operator(QFT(k)).data)))