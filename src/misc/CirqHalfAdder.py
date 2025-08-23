""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# CirqHalfAdder.py:     Cirq circuit for half adder

import cirq

q0, q1, q2 = cirq.LineQubit.range(3)    # Create 3 qubits
circuit = cirq.Circuit()                # Build circuit
circuit.append(cirq.X(q0))              # Append X to q0
circuit.append(cirq.X(q1))              # Append X to q1
circuit.append(cirq.TOFFOLI(q0, q1,q2)) # Append TOFFOLI to 3 qs
circuit.append(cirq.CNOT(q0, q1))       # Append CNOT to q0 & q1
print(circuit)                          # Output circuit
s = cirq.Simulator()                    # Initialize Simulator
print('Simulate the circuit:')
results = s.simulate(circuit)           # Run simulator
print(results)
"""---------------Output----------------
Simulate the circuit
output vector:                 |101>"""