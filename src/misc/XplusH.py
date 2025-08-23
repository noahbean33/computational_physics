""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# XplusH.py: Cirque program, setup X & H gates and |1>

import cirq

circuit = cirq.Circuit()              # Build circuit
a = cirq.NamedQubit('a')              # Define qubit
circuit.append(cirq.X(a))             # Append X (NOT) gate
circuit.append(cirq.H(a))             # Append H gate
s = cirq.Simulator()                  # Initialize simulator
print('Simulate the circuit:')
print(circuit)
results = s.simulate(circuit)         # Run simulator
print(results)
"""-------Cirque Output----------------------------------
Simulate the circuit:use
a:_______X_____H______
output vector:                   0.707|0> - 0.707|1>"""