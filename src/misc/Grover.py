""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code by G. He.), R Landau, Oregon State Unv, MJ Paez, 
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""                    

# Grover.py: Driver code for QC Grover algorithm on simulator & IBM Quantum

def oracle(omega: int) -> Gate:
      # Flip the sign if state is |omega>
      if omega < 0 or omega >= 16:
          raise ValueError("Input should be between "
                           + "0 and 15, got", omega)
      # Convert omega into bit pattern
      bit_string = f"{omega:04b}"
      # Start a quantum circuit of 4 qubits
      quantum_circuit = QuantumCircuit(4)
      [quantum_circuit.x(3-idx)
       for idx in range(4) if bit_string[idx] == '0']
      quantum_circuit.h(3)
      quantum_circuit.mcx([0, 1, 2], 3)
      quantum_circuit.h(3)
      [quantum_circuit.x(3-idx)
       for idx in range(4) if bit_string[idx] == '0']
      cap_u_omega = quantum_circuit.to_gate()
      cap_u_omega.name = "$U_\\omega$"
      return cap_u_omega
      
def diffuser(n_qubits: int) -> Gate:
      # Performs I-2|psi><psi|
      # Where |psi> is the uniform superposition state
      # Create a circuit with n_qubits
      quantum_circuit = QuantumCircuit(n_qubits)
      # Map |psi> to |0...0>
      quantum_circuit.h(range(n_qubits))
      # Map |0...0> to |1...1>
      quantum_circuit.x(range(n_qubits))
      # Multiple controlled-z
      # To flip sign for |1...1>
      quantum_circuit.h(n_qubits-1)
      quantum_circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
      quantum_circuit.h(n_qubits-1)
      # Map back from |1...1> to |0...0>
      quantum_circuit.x(range(n_qubits))
      # Map back |0...0> to |psi>
      quantum_circuit.h(range(n_qubits))
      cap_u_psi = quantum_circuit.to_gate()
      cap_u_psi.name = "$U_{\\psi}$"
      return cap_u_psi
      
if __name__ == "__main__":
      cap_n = 4
      qc = QuantumCircuit(cap_n)
      qc.h(range(cap_n))  # put into |\psi>
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
      plot_histogram(histogram, filename="grover4_sim_histogram.png")
      print(max(histogram, key=histogram.get))
      # Load account and get provider
      provider = IBMProvider(instance="ibm-q/open/main")
      device = least_busy(provider.backends(
          filters=lambda x: int(x.configuration().n_qubits) >= cap_n
                            and not x.configuration().simulator
                            and x.status().operational is True))
      print("Running on least busy device: ", device)
      # Transpile and run
      transpiled_circuit = transpile(qc, device)
      job = device.run(transpiled_circuit)
      job_monitor(job, interval=2)
      # Get result
      result = job.result()
      histogram = result.get_counts(qc)
      plot_histogram(histogram, filename="grover4_histogram.png")
      