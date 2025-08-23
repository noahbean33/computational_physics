""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# TwoHgates.py: Cirq program to create 2 H gates on one line

import cirq

circuit = cirq.Circuit()          # Build circuit
a = cirq.NamedQubit('a')          # Define named qubit
circuit.append(cirq.H(a))         # Append H gate to a
circuit.append(cirq.H(a))         # Append another H gate to a
s = cirq.Simulator()              # Initialize simulator
print('Simulate the circuit:')
print(circuit)                    # Output circuit
results = s.simulate(circuit)     # Run simulator
print(results)                    # Output resulting kets
"""--------------------Cirque's Output---------------------
 Simulate the circuit:
                          a: _______ H________ H_______
output vector:                           |0>"""