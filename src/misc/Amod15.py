""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright (this code by G. He.), R Landau, Oregon State Unv, MJ Paez, 
    Univ Antioquia, C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""                    

# amod15.py: Quantum circuit for Shor's factoring algorithm

  def amod15(a_in: int, p_in: int) -> QuantumCircuit:
      # Multiplication by a_in mod 15
      if a_in not in [2, 4, 7, 8, 11, 13, 14]:
          raise ValueError("'a_in' must be 2,4,7,8,11,13 or 14")
      quantum_circuit = QuantumCircuit(4)
      for iteration in range(p_in):
          if a_in in [2, 13]:
              quantum_circuit.swap(2, 3)
              quantum_circuit.swap(1, 2)
              quantum_circuit.swap(0, 1)
          if a_in in [7, 8]:
              quantum_circuit.swap(0, 1)
              quantum_circuit.swap(1, 2)
              quantum_circuit.swap(2, 3)
          if a_in in [4, 11]:
              quantum_circuit.swap(1, 3)
              quantum_circuit.swap(0, 2)
          if a_in in [7, 11, 13, 14]:  # I added 14 here
              for i in range(4):
                  quantum_circuit.x(i)
      quantum_circuit.name = "%i^%i mod 15" % (a_in, p_in)
      return quantum_circuit                # Return the circuit