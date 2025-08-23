""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
#  CirToffoli.py: Cirq program with 3 qubit CCNOT gate

import cirq

q0, q1, q2 = cirq.LineQubit.range(3)    # Create 3 qubits
circuit = cirq.Circuit()                # Build circuit
circuit.append(cirq.X(q0))              # Append X to q0
circuit.append(cirq.Z(q2))              # Append Z to q2
circuit.append(cirq.TOFFOLI (q0, q1,q2)) # Connect all 3 wi TOFFOLI print(circuit)                          # Output circuit
s = cirq.Simulator()                    # Initialize Simulator
print('Simulate the circuit:')
results = s.simulate(circuit)           # Run simulator
print(results)
"""-----------Output---------------------
Simulate the circuit:
output vector:                   |101>"""