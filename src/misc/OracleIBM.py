""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code by G He), R Landau, Oregon State Unv, MJ Paez,
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# OracleIBM.py: IBM QC Qiskit code for Oracle circuit, i=0-15.

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit.tools import job_monitor
from qiskit_ibm_provider import IBMProvider, least_busy
from numpy import math

def oracle(omega: int):  # remove ->Gate
    # Flip the sign if state is|omega>
    if omega < 0 or omega >= 16:
        raise ValueError("Need input" + "0 - 15, got ", omega)
    # Convert omega into bit pattern
    # bit_string = f"{omega:   04b}"  ########################
    bit_string = f"{omega:04b}"
    # print("bit-string",bit_string)
    # Start a  quantum circuit of 4 qubits
    quantum_circuit = QuantumCircuit(4)
    [quantum_circuit.x(3 - idx) for idx in range(4) if bit_string[idx] == '0']
    quantum_circuit.h(3)
    quantum_circuit.mcx([0, 1, 2], 3)
    quantum_circuit.h(3)
    [quantum_circuit.x(3 - idx) for idx in range(4) if bit_string[idx] == '0']
    cap_u_omega = quantum_circuit.to_gate()
    cap_u_omega.name = "$U_\\omega$"   
    return cap_u_omega
    
# diffuser.py: a quantum circuit for a general diffuser with n qubits
def diffuser(n_qubits: int):  # remove Gate
    # Where|psi>is the uniform superposition state
    # Create a circuit with n_qubits
    quantum_circuit = QuantumCircuit(n_qubits)
    # Map|psi> to |0...0>
    quantum_circuit.h(range(n_qubits))
    # Map|0...0> to |1...1>
    quantum_circuit.x(range(n_qubits))
    # Multiply controlled-z
    # To flip sign for |1...1>
    quantum_circuit.h(n_qubits - 1)
    quantum_circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    quantum_circuit.h(n_qubits - 1)
    # Map back from |1...1> to |0...0>
    quantum_circuit.x(range(n_qubits))
    # Map back |0...0> to |psi>
    quantum_circuit.h(range(n_qubits))
    cap_u_psi = quantum_circuit.to_gate()
    cap_u_psi.name = "$U_{\\psi}$"
    return cap_u_psi

# Grover.py:Driver code for QCGrover algorithm on simulator&IBMQuantum
if __name__ == "__main__":
    # these 2 commented lines only need to run once at the very beginning 
    # token ='***********'
    # QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
    cap_n = 4  # number of qubits
    qc = QuantumCircuit(cap_n)
    qc.h(range(cap_n))  # put into|\psi>
    # Run Grover iteration for R times
    cap_r = math.ceil(math.pi * math.sqrt(cap_n) / 4)
    for i in range(cap_r):
        qc.append(oracle(9), range(cap_n))
        qc.append(diffuser(cap_n), range(cap_n))
    qc.measure_all()
    qc.draw(output="mpl", filename="grover4_circuit.png")
    # Run on simulator
    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(qc, backend=backend)
    job = backend.run(transpiled_circuit)
    result = job.result()
    histogram = result.get_counts()
    plot_histogram(histogram, figsize=(7, 7), filename="grover4_sim_histogram.png")
    print(max(histogram, key=histogram.get))
    # Load account and get provider
    provider = IBMProvider(instance="ibm-q/open/main")
    device = least_busy(provider.backends(
            filters=lambda x: int(x.configuration().n_qubits) >= cap_n
            and not x.configuration().simulator
            and x.status().operational is True))
    print("Running on least busy device:", device)
    # Transpile and run
    transpiled_circuit = transpile(qc, device)
    job = device.run(transpiled_circuit)
    job_monitor(job, interval=2)
    # Get result
    result = job.result()
    histogram = result.get_counts(qc)
    plot_histogram(histogram, figsize=(7, 7), filename="grover4_histogram.png")