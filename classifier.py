import TensorNetwork as TN
from qiskit import QuantumCircuit, Aer, execute

sv_backend = Aer.get_backend("statevector_simulator")
qasm_backend = Aer.get_backend("qasm_simulator")


class QuantumClassifier:
    """
    Quantum neural network that takes a state and returns a label.
    Essentially this is a wrapper for a quantum circuit
    and maybe a measurement in the end
    """

    def __init__(self, tn: TN.TensorNetwork):
        self.TN = tn


    def binary_label(self, state_circ: QuantumCircuit, shots=1024) -> float:
        """
        Determines which class does a state belong to
        :param state_circ: circuit preparing the input state
        :return: 0 if the states belongs to the class 'L', 1 -- if to 'R',
        and a continuum of intermediate values
        """
        tn_circ = self.TN.construct_circuit()
        total_circ = state_circ + tn_circ
        job = execute(total_circ, qasm_backend, shots=shots)
        result = job.result()
        answer = result.get_counts()
        label = 0
        for key, value in answer.items():
            label += value * key.count('1')
        label = label / self.TN.n_qubits / shots
        return label
