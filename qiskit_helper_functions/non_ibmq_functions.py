import math, random, pickle, os, copy, random
from qiskit import QuantumCircuit, execute
from qiskit.providers import aer
from qiskit.circuit.classicalregister import ClassicalRegister
import qiskit.circuit.library as library
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.compiler import transpile
import numpy as np
import psutil

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore, gen_adder, gen_grover
from qiskit_helper_functions.conversions import dict_to_array

def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest

def read_dict(filename):
    if os.path.isfile(filename):
        f = open(filename,'rb')
        file_content = {}
        while 1:
            try:
                file_content.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
    else:
        file_content = {}
    return file_content

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1

def apply_measurement(circuit,qubits):
    measured_circuit = QuantumCircuit(circuit.num_qubits, len(qubits))
    for circuit_inst, circuit_qubits, circuit_clbits in circuit.data:
        measured_circuit.append(circuit_inst,circuit_qubits,circuit_clbits)
    measured_circuit.barrier(qubits)
    measured_circuit.measure(qubits,measured_circuit.clbits)
    return measured_circuit

def generate_circ(num_qubits,depth,circuit_type):
    def gen_secret(num_qubit):
        num_digit = num_qubit-1
        num = 2**num_digit-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros
    
    if not (num_qubits%2==0 and num_qubits>2):
        full_circ = None
    else:
        i,j = factor_int(num_qubits)
        if circuit_type == 'supremacy_linear':
            full_circ = gen_supremacy(1,num_qubits,depth,regname='q')
        elif circuit_type == 'supremacy':
            if abs(i-j)<=2:
                full_circ = gen_supremacy(i,j,depth,regname='q')
            else:
                full_circ = None
        elif circuit_type == 'hwea':
            full_circ = gen_hwea(i*j,depth,regname='q')
        elif circuit_type == 'bv':
            full_circ = gen_BV(gen_secret(i*j),barriers=False,regname='q')
        elif circuit_type == 'qft':
            full_circ = library.QFT(num_qubits=num_qubits,approximation_degree=0,do_swaps=False).decompose()
        elif circuit_type=='aqft':
            approximation_degree=int(math.log(num_qubits,2)+2)
            full_circ = library.QFT(num_qubits=num_qubits,approximation_degree=num_qubits-approximation_degree,do_swaps=False).decompose()
        elif circuit_type == 'sycamore':
            full_circ = gen_sycamore(i,j,depth,regname='q')
        elif circuit_type == 'adder':
            full_circ = gen_adder(nbits=int((num_qubits-2)/2),barriers=False,regname='q')
        elif circuit_type == 'grover':
            full_circ = gen_grover(width=num_qubits)
        else:
            raise Exception('Illegal circuit type:',circuit_type)
    assert full_circ is None or full_circ.num_qubits==num_qubits
    return full_circ

def find_process_jobs(jobs,rank,num_workers):
    count = int(len(jobs)/num_workers)
    remainder = len(jobs) % num_workers
    if rank<remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs

def evaluate_circ(circuit, backend, options=None):
    circuit = copy.deepcopy(circuit)
    max_memory_mb = psutil.virtual_memory().total>>20
    max_memory_mb = int(max_memory_mb/4*3)
    simulator = aer.Aer.get_backend('aer_simulator',max_memory_mb=max_memory_mb)
    if backend=='statevector_simulator':
        circuit.save_statevector()
        result = simulator.run(circuit).result()
        counts = result.get_counts(0)
        prob_vector = np.zeros(2**circuit.num_qubits)
        for binary_state in counts:
            state = int(binary_state,2)
            prob_vector[state] = counts[binary_state]
        return prob_vector
    elif backend == 'noiseless_qasm_simulator':
        if isinstance(options,dict) and 'num_shots' in options:
            num_shots = options['num_shots']
        else:
            num_shots = max(1024,2**circuit.num_qubits)

        if isinstance(options,dict) and 'memory' in options:
            memory = options['memory']
        else:
            memory = False
        if circuit.num_clbits == 0:
            circuit.measure_all()
        result = simulator.run(circuit, shots=num_shots, memory=memory).result()

        if memory:
            qasm_memory = np.array(result.get_memory(circuit))
            assert len(qasm_memory)==num_shots
            return qasm_memory
        else:
            noiseless_counts = result.get_counts(circuit)
            assert sum(noiseless_counts.values())==num_shots
            noiseless_counts = dict_to_array(distribution_dict=noiseless_counts,force_prob=True)
            return noiseless_counts
    else:
        raise NotImplementedError

def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name!='barrier':
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def dag_stripping(dag, max_gates):
    '''
    Remove all single qubit gates and barriers in the DAG
    Only leaves the first max_gates gates
    If max_gates is None, do all gates
    '''
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(dag.qregs[qreg_name]) for qreg_name in dag.qregs]
    vertex_added = 0
    for vertex in dag.topological_op_nodes():
        within_gate_count = max_gates is None or vertex_added<max_gates
        if vertex.op.name!='barrier' and len(vertex.qargs)==2 and within_gate_count:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
            vertex_added += 1
    return stripped_dag
