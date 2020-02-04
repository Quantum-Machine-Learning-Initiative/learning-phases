from qiskit import QuantumCircuit
import numpy as np

def GHZ(q, c=None):
    '''Returns the circuit implementing the Greenberger-Horne-Zeilinger
    state'''
    if c is not None:
        circ = QuantumCircuit(q, c)
    else:
        circ = QuantumCircuit(q)
    circ.h(q[0])
    for i in range(len(q) - 1):
        circ.cx(q[i], q[i+1])

    return circ

def random_clifford(q, c=None, depth=10):
    '''build a random Clifford circuit'''
    
    if c is not None:
        circ = QuantumCircuit(q, c)
    else:
        circ = QuantumCircuit(q)

    p_h = 0.2
    p_phase = 0.2
    
    for d in range(depth):
        qubit_1 = np.random.randint(len(q))
        randval = np.random.ranf()
        if randval < p_h:
            circ.h(q[qubit_1])
        elif randval < p_h + p_phase:
            circ.s(q[qubit_1])
        else:
            qubit_2 = np.random.randint(len(q)-1)
            if qubit_2 >= qubit_1:
                qubit_2 += 1
            circ.cx(q[qubit_1], q[qubit_2])
    return circ

