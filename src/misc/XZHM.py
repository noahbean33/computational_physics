""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
    
# XZHM.py: Cirq Simulation  with H, X, Z Gates + Measurement Op

import cirq

import matplotlib.pyplot as plt
circuit = cirq.Circuit()            # Build circuit
a = cirq.NamedQubit('a')            # Create qubit
circuit.append(cirq.X(a))           # Append X gate
circuit.append(cirq.Z(a))           # Append Z gate
circuit.append(cirq.H(a))           # Append H gate
s = cirq.Simulator()                # Initialize simulator
print('Simulate the circuit:')
results = s.simulate(circuit)       # Run simulator
circuit.append(cirq.measure(a, key ='result'))
samples = s.run(circuit,repetitions =1000)
print(circuit)
print(resultsC
print(samples)
cirq.plot_state_histogram(samples)
plt.show()
"""---------------Cirq Output--------------
Simulate the circuit:
a:                    ______X_______Z______H_______M____________
output vector:                      -0.707|0>  +  0.707|1>
result=1001110001001101001110111000100001...."""