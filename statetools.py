import numpy as np

def psi_to_rho(psi):
    '''Turns a pure state into a density matrix'''
    return np.outer(psi, np.array(psi).conj())

def state_partition(psi, m=1):
    '''Partitions a state into the first m qubits and the rest,
    returns the reduced density matrix'''
    rho = psi_to_rho(psi)
    n_qubits = round(np.log(np.shape(rho)[0]) / np.log(2))
    tensor_shape = [2] * n_qubits * 2
    rho = rho.reshape(tensor_shape)
    if m >= n_qubits or m==0:
        raise ValueError('Invalid separation line')
    qubits_to_contract = n_qubits - m
    for i in range(qubits_to_contract):
        rho = np.trace(rho,
                       axis1=(n_qubits - 1 - i),
                       axis2=(2 * n_qubits - 2 - 2 * i))
    return rho
    

def even_power_invariant(rho, k):
    '''Sums the singular values raised to the power 2k'''
    s = svdvals(rho)
    return np.sum(s**(2 * k))
