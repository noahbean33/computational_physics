""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# CirqCNOT.py:  Cirq program with CNOT gate

import cirq

circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2)        # Create two qubits
circuit.append(cirq.X(q0))              # Append X to q0
circuit.append(cirq.Z(q1))              # Append Z to q1
circuit.append(cirq.CNOT(q0, q1))       # Append CNOT, q0 = control
print(circuit)
s = cirq.Simulator()                    # Initialize Simulator
print('Simulate the circuit:')
results = s.simulate(circuit)           # Run simulator
print(results)
"""-------------------Output---------------------
Simulate the circuit:
measurements: (no measurements)
output vector:                       |11>"""