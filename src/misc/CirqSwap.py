""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# CirqSwap.py: Cirq program to create & swap 2 qubits

import cirq

circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2)         # Create two qubits
circuit.append(cirq.X(q0))               # Append X to q0
circuit.append(cirq.Z(q1))               # Append Z to q1
circuit.append(cirq.SWAP(q0,q1))         # Swap qubits
print(circuit)
s = cirq.Simulator()                     # Initialize simulator
print('Simulate the circuit:')
results = s.simulate(circuit)            # Run simulator
print(results)
"""----------Cirque Output ----------------------------
Simulate the circuit:
output vector:                     |01>"""