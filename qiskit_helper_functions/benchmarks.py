import qiskit.circuit.library as library
import math, qiskit, random, logging
import networkx as nx
import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit

from qcg.generators import (
    gen_supremacy,
    gen_hwea,
    gen_BV,
    gen_sycamore,
    gen_adder,
)

# from local_qiskit_helper_functions.random_benchmark import RandomCircuit


def factor_int(n: int) -> Tuple[int, int]:
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n / val)
        if val * co_val == n:
            break
        else:
            val -= 1
    return val, co_val


def gen_secret(num_qubit):
    num_digit = num_qubit - 1
    num = 2**num_digit - 1
    num = bin(num)[2:]
    num_with_zeros = str(num).zfill(num_digit)
    return num_with_zeros


def construct_qaoa_plus(P, G, params, reg_name):
    assert len(params) == 2 * P, "Number of parameters should be 2P"

    nq = len(G.nodes())
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nq, reg_name))

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

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    return circ


def construct_random(num_qubits, depth):
    random_circuit_obj = RandomCircuit(
        width=num_qubits, depth=depth, connection_degree=0.5, num_hadamards=5, seed=None
    )
    circuit, _ = random_circuit_obj.generate()
    return circuit


def ghz(num_qubits: int):
    """
    GHZ state
    https://qiskit.org/documentation/stable/0.24/tutorials/noise/9_entanglement_verification.html
    """
    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.h(0)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def w_state(num_qubits: int):
    """
    W state
    https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/awards/teach_me_qiskit_2018/w_state/W%20State%201%20-%20Multi-Qubit%20Systems.ipynb
    Single qubit gates are proxies only.
    Circuit structure is accurate.
    """

    def F_gate(circ, i, j, n, k):
        theta = np.arccos(np.sqrt(1 / (n - k + 1)))
        circ.ry(-theta, j)
        circ.cz(i, j)
        circ.ry(theta, j)

    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.x(num_qubits - 1)
    for qubit in range(num_qubits - 1, -1, -1):
        F_gate(circuit, qubit, qubit - 1, num_qubits, num_qubits - qubit)
    for qubit in range(num_qubits - 2, -1, -1):
        circuit.cx(qubit, qubit + 1)
    return circuit


def generate_circuit(
    circuit_type: str, num_qubits: int, depth: int, connected_only=True
) -> QuantumCircuit:
    """
    Generate a circuit
    connected_only: only return a circuit with one tensor component.
    If the setup combination is available for a circuit type, assert corresponding errors.
    """
    match circuit_type:
        case "supremacy":
            grid_length = int(math.sqrt(num_qubits))
            assert (
                grid_length**2 == num_qubits
            ), "Supremacy is defined for n*n square circuits"
            circuit = gen_supremacy(
                grid_length, grid_length, depth * 8, order="random", regname="q"
            )
        case "sycamore":
            grid_length = int(math.sqrt(num_qubits))
            assert (
                grid_length**2 == num_qubits
            ), "Sycamore is defined for n*n square circuits"
            circuit = gen_sycamore(grid_length, grid_length, depth * 8, regname="q")
        case "qft":
            circuit = library.QFT(
                num_qubits=num_qubits, approximation_degree=0, do_swaps=False
            ).decompose()
        case "aqft":
            approximation_degree = int(math.log(num_qubits, 2) + 2)
            circuit = library.QFT(
                num_qubits=num_qubits,
                approximation_degree=num_qubits - approximation_degree,
                do_swaps=False,
            ).decompose()
        case "regular":
            assert (
                3 * num_qubits % 2 == 0
            ), "3-regular graph must have even number of qubits"
            graph = nx.random_regular_graph(3, num_qubits)
            circuit = construct_qaoa_plus(
                P=depth,
                G=graph,
                params=[np.random.uniform(-np.pi, np.pi) for _ in range(2 * depth)],
                reg_name="q",
            )
        case "erdos":
            random_density = np.random.uniform(0, 1)
            graph = nx.generators.random_graphs.erdos_renyi_graph(
                num_qubits, random_density
            )
            if connected_only:
                density_delta = 0.001
                lo = 0
                hi = 1
                while lo < hi:
                    mid = (lo + hi) / 2
                    graph = nx.generators.random_graphs.erdos_renyi_graph(
                        num_qubits, mid
                    )
                    if nx.number_connected_components(graph) == 1:
                        hi = mid - density_delta
                    else:
                        lo = mid + density_delta
            circuit = construct_qaoa_plus(
                P=depth,
                G=graph,
                params=[np.random.uniform(-np.pi, np.pi) for _ in range(2 * depth)],
                reg_name="q",
            )
        case "ghz":
            circuit = ghz(num_qubits=num_qubits)
        case "wstate":
            circuit = w_state(num_qubits=num_qubits)
        case "random":
            circuit = construct_random(num_qubits=num_qubits, depth=depth)
        case "bv":
            circuit = gen_BV(gen_secret(num_qubits), barriers=False, regname="q")
        case _:
            raise Exception("{:s} is not implemented".format(circuit_type))
    if connected_only and circuit.num_tensor_factors() != 1:
        raise ValueError("Benchmark circuit is not connected")
    return circuit
