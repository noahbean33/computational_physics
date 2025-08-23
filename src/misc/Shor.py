""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright G. He, R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# Shor.py: Shor's algorithm
# https://qiskit.org/textbook/ch-algorithms/shor.html

import random
from fractions import Fraction
from math import gcd
from typing import List
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit.library import QFT

def amod15(a_in: int, p_in: int) -> QuantumCircuit: # Mult x a_in mod 15
    if a_in not in [2, 4, 7, 8, 11, 13, 14]:
        raise ValueError("'a_in' must be 2,4,7,8,11,13 or 14")
    quantum_circuit = QuantumCircuit(4)
    for iteration in range(p_in):
        if a_in in [2, 13]:
            quantum_circuit.swap(2, 3)
            quantum_circuit.swap(1, 2)
            quantum_circuit.swap(0, 1)
        if a_in in [7, 8]:
            quantum_circuit.swap(0, 1)
            quantum_circuit.swap(1, 2)
            quantum_circuit.swap(2, 3)
        if a_in in [4, 11]:
            quantum_circuit.swap(1, 3)
            quantum_circuit.swap(0, 2)
        if a_in in [7, 11, 13, 14]:  # I added 14 here
            for i in range(4):
                quantum_circuit.x(i)
    quantum_circuit.name = "%i^%i mod 15" % (a_in, p_in)
    return quantum_circuit  # return the circuit
    
def qpe(u_list: List[QuantumCircuit]) -> float:  # Build phase circuit
    # u_list: a list of QuantumCircuit
    # [U^(2^0), U^(2^1), ... U^(2^(t-1))]
    t = len(u_list)
    num_qubits_u = u_list[0].num_qubits        # N qubits for cap_u gate
    qc = QuantumCircuit(t + num_qubits_u, t)
    # put the first t_count qubits into superposition
    for i in range(t):
        qc.h(i)
    # put the last n_u qubit into |1> state
    qc.x(t)			                           # qiskit convention
    for i in range(t):                         # Add contr-U^{2^j} gate
        qc.append(u_list[i].to_gate().control(), [i]
                  + [j+t for j in range(num_qubits_u)])
    qc.append(QFT(t, inverse=True).to_gate(), range(t))  # Inverse QFT
    qc.measure(range(t), range(t))			        # Finally, measure
    simulator = Aer.get_backend("aer_simulator")    # Run on simulator
    q_obj = assemble(transpile(qc, simulator), shots=1)
    result = simulator.run(q_obj, memory=True).result()
    readings = result.get_memory()
    print("Register reading: " + readings[0])
    phase = int(readings[0], 2)/(2**t)
    print("Corresponding phase: %f" % phase)
    return phase

if __name__ == "__main__":
    cap_n = 15
    factor_found = False
    attempt = 0
    while not factor_found:
        print("Attempt #", attempt)
        attempt += 1
        a = random.randint(2, cap_n-1)
        print("Random a = ", a)
        k = gcd(a, cap_n)
        if k != 1:
            factor_found = 1
            print("Found factor: ", k)
        else:
            p = qpe([amod15(a, 2**j) for j in range(8)])
            print("Phase: ", p)
            fraction = Fraction(p).limit_denominator(cap_n)
            s, r = fraction.numerator, fraction.denominator
            print("r = ", r)
            if r % 2 == 0:  # r is even
                guesses = [gcd(a**(r//2)+1, cap_n),
                           gcd(a**(r//2)-1, cap_n)]
                for g in guesses:
                    if g not in [1, cap_n] and (cap_n % g) == 0:
                        print("Found factor: %i" % g)
                        factor_found = True