""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# FullAdder.py: Cirq q0+q1 full adder program

import cirq

circuit = cirq.Circuit()# Build circuit
q0, q1, q2,q3 = cirq.LineQubit.range(4)     # Create 4 qubits
circuit = cirq.Circuit()                    # Build circuit with qubits
circuit.append(cirq.X(q0))                  # Append X to q0
circuit.append(cirq.X(q1))                  # Append X to q1
circuit.append(cirq.TOFFOLI(q0, q1,q2))     # Append TOFFOLI
circuit.append(cirq.CNOT(q0, q1))           # Append CNOT to q0,q1
circuit.append(cirq.TOFFOLI(q1, q2,q3))     # Append TOFFOLI
circuit.append(cirq.CNOT(q1, q2))           # Append CNOT to q1,q2
circuit.append(cirq.CNOT(q0, q1))           # Append CNOT to q0,q1
print(circuit)
s = cirq.Simulator()                        # Initialize Simulator
print('Simulate the circuit:')
results = s.simulate(circuit)               # Run simulator
print(results)
"""------------Output----------------------------
Simulate the circuit:
output vector:           |1110>"""