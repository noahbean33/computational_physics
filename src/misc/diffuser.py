""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code by G. He.), R Landau, Oregon State Unv, MJ Paez, 
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""                    

# diffuser.py: a quantum circuit for a general diffuser with n qubits

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