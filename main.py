import warnings
import sys
import copy

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from math import pi
from qiskit import Aer, execute
from qiskit.tools.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from scipy.optimize import minimize
import time
import json
import uuid

from qiskit_aqua import Operator
from qiskit_aqua.algorithms import VQE
from qiskit_aqua.components.variational_forms import RYRZ
from qiskit_aqua.components.optimizers import L_BFGS_B

import Entangler
import TensorNetwork
import TNOptimize
import hamiltonians
import utils
import circuits


sv_backend = Aer.get_backend("statevector_simulator")


def visualize_circuit():
    n_qubits = 8
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    e = Entangler.DumbEntangler()
    TN = TensorNetwork.TreeTN(q, c, e)
    circ = TN.construct_circuit()
    print(circ)


def jsonize(xk):
    '''Turn a numpy array of complex numbers
    into a list of pairs of floats'''
    return [[z.real, z.imag] for z in xk]


def save_TN_state(TN, vals, E, params_order, comment):
    sol_id = str(uuid.uuid4())
    circ = TN.construct_circuit(vals)
    circ_qasm = circ.qasm()
    
    job = execute(circ, sv_backend)
    result = job.result()
    state = result.get_statevector(circ)
    state_jsonable = [[z.real, z.imag] for z in state]


    solution = {'id': sol_id,
                'state': state_jsonable,
                'circ': circ_qasm,
                'H': ham_dict,
                'E': E,
                'values': vals,
                'n_params': TN.n_params,
                'params_order': params_order,
                'comment': comment}

    with open(('saved_states/TN_state_'
               + time.strftime("%Y-%m-%d_%H-%M-%S")
               + '_'
               + sol_id[:6]
               + '.json'), 'w') as out_file:
        json.dump(solution, out_file, indent=2)    





def load_TN_state(filename):
    with open(filename, 'r') as in_file:
        solution = json.load(in_file)
    return solution


def my_callback(xk):
    global n_iters
    global f
    n_iters += 1
    print("iteration number: ", n_iters, ", f = {0:0.4f}".format(f(xk)))


if __name__ == '__main__':

    # ns = list(range(2, 10))
    # #ns = [3]
    # entropy = []
    # h_list = np.linspace(0, 2, num=11)
    # n_qubits = 8
    # ham_dict = hamiltonians.ising_model(n_qubits, 1, 1)
    # multiplicity, w, v = hamiltonians.exact_gs(ham_dict)
    #
    # for m in range(1, n_qubits):
    #     rho = utils.state_partition(v[:, 0], m=m)
    #     #print(rho)
    #     E = utils.get_entropy(rho)
    #     entropy.append(E)
    #     print(E)
    # plt.scatter(list(range(1, n_qubits)), entropy)
    # plt.show()





    ################ DEFINE BASIC ATTRIBUTES ##############
    n_qubits = 2

    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    ent = Entangler.IsingEntangler()
    method = 'L-BFGS-B'
    tol = 1e-6

    J = 1
    #h_list = np.linspace(0, 2, num=11)
    h_list = [1.4]

    depth = 2
    TN = TensorNetwork.Checkerboard(q, c, ent, depth=depth)
    # TN = TensorNetwork.RankOne(q, c)
    #np.random.seed(1)
    u = np.random.rand(TN.n_params)

    # u = [
    #     0.9928212637462062,
    #     0.7787765603478692,
    #     -0.6599714601284891,
    #     0.12774548793384877,
    #     1.8340264212444775,
    #     -0.11877431406314068,
    #     1.1024893903950908,
    #     0.6334137773344314,
    #     0.6889549457179684,
    #     0.6403775809314097,
    #     1.4557314032455315,
    #     0.7305950244399907,
    #     0.7735460111101621,
    #     0.750836359962941,
    #     0.6661832429670945,
    #     1.1484096692328203,
    #     1.6094357740153986,
    #     1.0828883263584101,
    #     -0.059362750361968135,
    #     0.8826976357461255,
    #     0.5614661989332314,
    #     0.322909041605473,
    #     0.2915130741893068,
    #     0.16380001077188414,
    #     0.2958040219477211,
    #     1.1506170040688564,
    #     0.8411133555339009,
    #     -0.5739914468926891,
    #     1.2128825415421658,
    #     1.2638544459639531,
    #     0.1287983784941154,
    #     0.8666214371036228,
    #     0.5179919664179069,
    #     0.49863025823372775,
    #     0.8826115701031134,
    #     1.0960487463725648,
    #     0.6221001395705705,
    #     0.25013857934681283,
    #     -0.0029989143709251154,
    #     1.982937052710126,
    #     -0.9028316609788544,
    #     1.1391824938974893,
    #     1.493245484244748,
    #     0.8522895561819116,
    #     0.5499948311277437,
    #     0.6894046971282751,
    #     0.03884704817918759,
    #     0.014086528160476777,
    #     -0.43840352020053897,
    #     0.18746601628605825,
    #     1.189522182264418,
    #     -1.3961111968251654,
    #     0.5356735338427704,
    #     -1.97598501148733,
    #     1.4356941647243675,
    #     0.39717504122480657,
    #     2.0323594363913817,
    #     -0.10185553032351075,
    #     1.1082144565906473,
    #     -0.5212585527543456,
    #     1.428761706575597,
    #     -0.10524452659675866,
    #     -1.156644020153627,
    #     -0.015189646218125801,
    #     1.5993001071164537,
    #     -0.4739550325099412,
    #     1.5201601268978344,
    #     1.4922823207111606,
    #     1.5014031893375157,
    #     -0.6263514515857077,
    #     -1.4280611512086216,
    #     -0.9695611616408045,
    #     -0.14618038336327258,
    #     0.7237730194737819,
    #     1.4734845436149444,
    #     0.03155388868009967,
    #     -0.398919410358472,
    #     -0.23339936234702452,
    #     0.7166787766860735,
    #     0.48766701081771685,
    #     1.5523801512242348,
    #     0.3587917060599196,
    #     -0.41127919771503674,
    #     0.8482149565326942,
    #     -0.0038091943560501773,
    #     -0.004752837462888118,
    #     0.03363158786986,
    #     1.6063632710667064,
    #     1.497180478624074,
    #     0.46774001654784103,
    #     1.4905583256180404,
    #     -1.261497607855093,
    #     0.32320217690005737,
    #     1.1430880297598405,
    #     -1.9732303923334835,
    #     -0.09814174283491969,
    #     1.0344563176155874,
    #     0.3516288456124625,
    #     1.6778305342562843,
    #     -0.13408065348081571,
    #     0.15090631193877116,
    #     -0.11925922059923152,
    #     -0.35224049394418433,
    #     0.6105182962791843,
    #     1.3423948166830695,
    #     1.2062240548895888,
    #     0.32447632817017974,
    #     3.506381786900491,
    #     0.26287486417050154,
    #     -0.0697807788284994,
    #     0.14460808773669837,
    #     0.22299534967330623,
    #     0.5878275070069225,
    #     0.3385599957087916,
    #     0.39895436160725506,
    #     -1.3823764006225592,
    #     1.4903142371041354,
    #     0.03713928928427975,
    #     3.1148938433205693,
    #     -0.08831768858576738,
    #     1.2716983422275754,
    #     0.0021867698238301012,
    #     0.3418129714254196,
    #     0.16689232103437612,
    #     0.5029101245903237
    #   ]

    # nz = TN.n_params - len(u)

    # u = u + [0] * nz

    # ydata = []
    #
    for h in h_list:
        print('''

        ------ h = {0:0.2f} -------

        '''.format(h))

        ham_dict = hamiltonians.ising_model(n_qubits, J, h)
        H = hamiltonians.explicit_hamiltonian(ham_dict)
        f = TNOptimize.build_objective_function(TN, explicit_H=H)
        
        
        
        n_iters = 0
        start_time = time.time()
        res = minimize(f, u, options={'maxiter': 300}, callback=my_callback, tol=tol, method=method)
        time_spent = time.time() - start_time
        print('Elapsed time: {0:5.2f}'.format(time_spent))
        
        circ = TN.construct_circuit(res.x)
        state = utils.get_state(circ)
        print(res.fun)
        # w, v = np.linalg.eigh(H)
        # state = v[:, 0]
        # E = w[0]
        # print(E)
        time.sleep(1.5)
        ############## PACK EVERYTHING INTO A JSON AND SAVE ##########
        # datetime = time.strftime("%Y-%m-%d_%H-%M-%S")

        # problem = {'title': 'Ising model',
        #            'n_qubits': n_qubits,
        #            'h': h,
        #            'J': J,
        #            'Hamiltonian': ham_dict}

        # solution = {'n_qubits': n_qubits,
        #             # 'TN': {'name': str(type(TN)),
        #             #        'depth': depth,
        #             #        'circuit': circ.qasm(),
        #             #        'n_params': TN.n_params,
        #             #        'params': res.x.tolist()},
        #             'E': E,
        #             'state': jsonize(state),
        #             # 'optimizer': {'method': method,
        #             #               'n_iters': res.nit,
        #             #               'message': str(res.message),
        #             #               'time': time_spent,
        #             #               'tol': tol}
        #             }

        # data_entry = {'id': str(uuid.uuid4()),
        #               'datetime': datetime,
        #               'problem': problem,
        #               'solution': solution,
        #               'comment': 'Exact solution'}

        # with open(('saved_states/TN_state_'
        #            + datetime
        #            + '.json'), 'w', encoding='utf-8') as out_file:
        #     json.dump(data_entry, out_file, indent=2)


    
    
    #tree_order = (1, 4, 5, 6, 8)
    ##tree_order = (3, 4, 5)
    #tree_params_order = [list(range(k * ent.n_params, (k + 1) * ent.n_params))
    #                     for k in tree_order]
    #full_params_order = [list(range(k * ent.n_params, (k + 1) * ent.n_params))
    #                    for k in range(TN.n_tensors)]
    #params_order = tree_params_order + full_params_order * 2
    #
    #h_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    #zero_sol = load_TN_state('saved_states/TN_state_2019-02-13_15-36-40_41f5f2.json')
    #
    #vals = copy.deepcopy(zero_sol['values'])
    #
    #ham_dict = hamiltonians.ising_model(n_qubits, 1, 0.2)
    #H = hamiltonians.explicit_hamiltonian(ham_dict)
    #
    ##for i in range(len(vals)):
    ##    if vals[i] > np.pi:
    ##        vals[i] -= np.pi
    #
    #print(TNOptimize.measure_ham_2(TN.construct_circuit(vals), explicit_H=H))
    #
    #for h in h_list:
    #        ham_dict = hamiltonians.ising_model(n_qubits, 1, h)
    #        H = hamiltonians.explicit_hamiltonian(ham_dict)
    #
    #        E, vals = TNOptimize.any_order_VQE_2(TN, params_order,
    #                                             explicit_H=H, n_calls=100, verbose=False,
    #                                             init_vals=vals)
    #
    #
    #comment = 'Running local optimizer ' + method + ' for a checkers with depth {}'.format(depth)
    #save_TN_state(TN, vals.tolist(), E, params_order=[list(range(TN.n_params))], comment=comment)
    #        print('Elapsed time: {0:5.2f}'.format(time.time() - start_time))
