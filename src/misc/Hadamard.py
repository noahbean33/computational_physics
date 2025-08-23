""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# Hadamard.py: Cirq program to create H gate

import cirq                             # Import Cirq

circuit = cirq.Circuit()                # Build circuit
qubit = cirq.GridQubit(0,0)             # Create qubit at (0,0)
circuit.append(cirq.H(qubit))           # Append Hadamard gate
s = cirq.Simulator()                    # Initialize simulator
print('Simulate the circuit:')
print(circuit)                          # Output circuit
results = s.simulate(circuit)           # Run simulator
print(results)                          # Output resulting kets
"""----------Cirque's Output ----------------------------------
Simulate the circuit:
(0,0):  __________H__________
output vector:                  0.707 |0> + 0.707|1>"""