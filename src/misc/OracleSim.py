""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code with G He), R Landau, Oregon State Unv, MJ Paez,
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024.
    Please respect copyright & acknowledge our work."""

# OracleSim.py: Simulator version of Qiskit code for Oracle circuit, i=0-15

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from numpy import math

def oracle(omega: int):                         # With removed->Gate
   if omega < 0 or omega >= 16:                # Flip sign if |omega>
       raise ValueError("Input should be betwn"+"0 & 15, got" , omega)
   bit_string =f"{omega: 04b}"         # Convert omega to bit pattern
   quantum_circuit = QuantumCircuit(4)         # 4 bit quantum circuit
   [quantum_circuit.x(3-idx) for idx in range(4) if bit_string[idx]=='0']
   quantum_circuit.h( 3 )
   quantum_circuit.mcx([0,1,2],3)
   quantum_circuit.h(3)
   [quantum_circuit.x( 3-idx) for idx in range(4) if bit_string[idx]=='0']
   cap_u_omega = quantum_circuit.to_gate()
   cap_u_omega.name="omega"                                # Differs
   return cap_u_omega
   
# diffuser.py: circuit for a diffuser with n qubits
def diffuser(n_qubits: int): # remove Gate, I - 22|psi><psi|
    quantum_circuit = QuantumCircuit(n_qubits)      # n qubit circuit
    quantum_circuit.h( range( n_qubits))         # Map|psi> to |0...0>
    quantum_circuit.x(range(n_qubits))         # Map|0...0> to |1...1>
    quantum_circuit.h( n_qubits -1) # Multi cntrl-z, flips sign |1...1>
    quantum_circuit.mcx(list(range(n_qubits -1)),n_qubits -1)
    quantum_circuit.h(n_qubits -1)               # Map back to |0...0>
    quantum_circuit.x(range( n_qubits))
    quantum_circuit.h( range( n_qubits))           # Map back to |psi>
    cap_u_psi = quantum_circuit.to_gate()
    cap_u_psi.name="$U_{\\psi}$"
    return cap_u_psi
    
# Grover.py: Driver code for QC Grover algorithm on simulator
if __name__ == "__main__" :
   cap_n=4
   qc = QuantumCircuit(cap_n)
   qc.h(range(cap_n))                         # put into|\psi>
   # Run 
   cap_r = math.ceil(math.pi*math.sqrt(cap_n)/4) # Iterate Grover R times
   for i in range(cap_r):
      qc.append(oracle(9), range(cap_n))
      qc.append(diffuser(cap_n), range(cap_n))
   qc.measure_all()
   qc.draw(output="mpl",filename="grover4_circuit.png")
   backend=Aer.get_backend("aer_simulator")            # Run on simulator
   transpiled_circuit = transpile(qc,backend=backend)
   job = backend.run(transpiled_circuit)
   result = job.result()
   histogram = result.get_counts()
   plot_histogram(histogram,filename="grover4_sim_histogram.png")		