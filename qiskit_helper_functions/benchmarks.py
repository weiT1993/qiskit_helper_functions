import qiskit.circuit.library as library
import math, qiskit, random
import networkx as nx
import numpy as np

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_sycamore, gen_adder
from qiskit_helper_functions.random_benchmark import RandomCircuit

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1

def gen_secret(num_qubit):
        num_digit = num_qubit-1
        num = 2**num_digit-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros

def construct_qaoa_plus(P, G, params, barriers=False, measure=False):
    assert len(params) == 2 * P, 'Number of parameters should be 2P'

    nq = len(G.nodes())
    circ = qiskit.QuantumCircuit(nq, name='q')

    # Initial state
    circ.h(range(nq))

    gammas = [param for i, param in enumerate(params) if i % 2 == 0]
    betas = [param for i, param in enumerate(params) if i % 2 == 1]
    for i in range(P):
        # Phase Separator Unitary
        for edge in G.edges():
            q_i, q_j = edge
            circ.rz(gammas[i] / 2, [q_i, q_j])
            circ.cx(q_i, q_j)
            circ.rz(-1 * gammas[i] / 2, q_j)
            circ.cx(q_i, q_j)
            if barriers:
                circ.barrier()

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    if measure:
        circ.measure_all()

    return circ

def construct_random(num_qubits):
    random_circuit_obj = RandomCircuit(width=num_qubits,depth=num_qubits,
    connection_degree=0.2,num_hadamards=5,seed=None)
    num_trials = 100
    while num_trials:
        circuit, _ = random_circuit_obj.generate()
        if circuit.num_tensor_factors()==1:
            return circuit
        else:
            num_trials -= 1
    return None

def generate_circ(num_qubits,depth,circuit_type,seed):
    random.seed(seed)
    full_circ = None
    i,j = factor_int(num_qubits)
    if circuit_type == 'supremacy':
        if abs(i-j)<=2:
            full_circ = gen_supremacy(i,j,depth*8,regname='q')
    elif circuit_type == 'sycamore':
        full_circ = gen_sycamore(i,j,depth,regname='q')
    elif circuit_type == 'hwea':
        full_circ = gen_hwea(i*j,depth,regname='q')
    elif circuit_type == 'bv':
        full_circ = gen_BV(gen_secret(i*j),barriers=False,regname='q')
    elif circuit_type == 'qft':
        full_circ = library.QFT(num_qubits=num_qubits,approximation_degree=0,do_swaps=False).decompose()
    elif circuit_type=='aqft':
        approximation_degree=int(math.log(num_qubits,2)+2)
        full_circ = library.QFT(num_qubits=num_qubits,approximation_degree=num_qubits-approximation_degree,do_swaps=False).decompose()
    elif circuit_type == 'adder':
        full_circ = gen_adder(nbits=int((num_qubits-2)/2),barriers=False,regname='q')
    elif circuit_type=='regular':
        if 3*num_qubits%2==0:
            num_trials = 100
            while num_trials:
                graph = nx.random_regular_graph(3, num_qubits)
                if nx.is_connected(graph):
                    full_circ = construct_qaoa_plus(P=depth,G=graph,
                    params=[np.random.uniform(-np.pi,np.pi) for _ in range(2*depth)])
                    break
                else:
                    num_trials -= 1
    elif circuit_type=='erdos':
        num_trials = 1000
        density = 0.001
        while num_trials:
            graph = nx.generators.random_graphs.erdos_renyi_graph(num_qubits, density)
            if nx.is_connected(graph):
                full_circ = construct_qaoa_plus(P=depth,G=graph,
                params=[np.random.uniform(-np.pi,np.pi) for _ in range(2*depth)])
                break
            else:
                num_trials -= 1
                density += 0.001
    elif circuit_type=='random':
        full_circ = construct_random(num_qubits)
    else:
        raise Exception('Illegal circuit type:',circuit_type)
    assert full_circ is None or full_circ.num_qubits==num_qubits
    return full_circ