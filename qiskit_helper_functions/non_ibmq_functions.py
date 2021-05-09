import math, random, pickle, os, copy
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.classicalregister import ClassicalRegister
import qiskit.circuit.library as library
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
import numpy as np

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore, gen_adder, gen_grover
from qiskit_helper_functions.conversions import dict_to_array

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
    meas = QuantumCircuit(circuit.num_qubits, len(qubits))
    meas.barrier(qubits)
    meas.measure(qubits,meas.clbits)
    qc = circuit+meas
    return qc

def generate_circ(full_circ_size,circuit_type):
    def gen_secret(num_qubit):
        num_digit = num_qubit-1
        num = 2**num_digit-1
        num = bin(num)[2:]
        num_with_zeros = str(num).zfill(num_digit)
        return num_with_zeros

    i,j = factor_int(full_circ_size)
    if circuit_type == 'supremacy_linear':
        full_circ = gen_supremacy(1,full_circ_size,8,regname='q')
    elif circuit_type == 'supremacy':
        if abs(i-j)<=2:
            full_circ = gen_supremacy(i,j,8,regname='q')
        else:
            full_circ = QuantumCircuit()
    elif circuit_type == 'hwea':
        full_circ = gen_hwea(i*j,1,regname='q')
    elif circuit_type == 'bv':
        full_circ = gen_BV(gen_secret(i*j),barriers=False,regname='q')
    elif circuit_type == 'qft':
        full_circ = library.QFT(num_qubits=full_circ_size,approximation_degree=0,do_swaps=False)
    elif circuit_type=='aqft':
        approximation_degree=int(math.log(full_circ_size,2)+2)
        full_circ = library.QFT(num_qubits=full_circ_size,approximation_degree=full_circ_size-approximation_degree,do_swaps=False)
    elif circuit_type == 'sycamore':
        full_circ = gen_sycamore(i,j,8,regname='q')
    elif circuit_type == 'adder':
        if full_circ_size%2==0 and full_circ_size>2:
            full_circ = gen_adder(nbits=int((full_circ_size-2)/2),barriers=False,regname='q')
        else:
            full_circ = QuantumCircuit()
    elif circuit_type == 'grover':
        if full_circ_size%2==0:
            full_circ = gen_grover(width=full_circ_size)
        else:
            full_circ = QuantumCircuit()
    elif circuit_type == 'random':
        full_circ = generate_random_circuit(num_qubits=full_circ_size,circuit_depth=20,density=0.5,inverse=True)
    else:
        raise Exception('Illegal circuit type:',circuit_type)
    assert full_circ.num_qubits==full_circ_size or full_circ.num_qubits==0
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
    if backend=='statevector_simulator':
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend=backend, optimization_level=0)
        result = job.result()
        output_sv = result.get_statevector(circuit)
        output_p = []
        for x in output_sv:
            amplitude = np.absolute(x)**2
            if amplitude>1e-16:
                output_p.append(amplitude)
            else:
                output_p.append(0)
        output_p = np.array(output_p)
        return output_p
    elif backend == 'noiseless_qasm_simulator':
        if isinstance(options,dict) and 'num_shots' in options:
            num_shots = options['num_shots']
        else:
            num_shots = max(1024,2**circuit.num_qubits)
        backend = Aer.get_backend('qasm_simulator')

        if isinstance(options,dict) and 'memory' in options:
            memory = options['memory']
        else:
            memory = False
        if circuit.num_clbits == 0:
            circuit = apply_measurement(circuit=circuit,qubits=circuit.qubits)
        noiseless_qasm_result = execute(circuit, backend, shots=num_shots, memory=memory).result()

        if memory:
            qasm_memory = np.array(noiseless_qasm_result.get_memory(0))
            assert len(qasm_memory)==num_shots
            return qasm_memory
        else:
            noiseless_counts = noiseless_qasm_result.get_counts(0)
            assert sum(noiseless_counts.values())==num_shots
            noiseless_counts = dict_to_array(distribution_dict=noiseless_counts,force_prob=True)
            return noiseless_counts
    elif backend=='noisy_qasm_simulator':
        noisy_qasm_result = execute(circuit, Aer.get_backend('qasm_simulator'),
        coupling_map=options['coupling_map'],
        basis_gates=options['basis_gates'],
        noise_model=options['noise_model'],
        shots=options['num_shots']).result()

        noisy_counts = noisy_qasm_result.get_counts(0)
        assert sum(noisy_counts.values())==options['num_shots']
        noisy_counts = dict_to_array(distribution_dict=noisy_counts,force_prob=True)
        return noisy_counts
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

def dag_stripping(dag):
    # Remove all single qubit gates and barriers in the DAG
    stripped_dag = copy.deepcopy(dag)
    for vertex in stripped_dag.topological_op_nodes():
        if len(vertex.qargs) != 2 or vertex.op.name=='barrier':
            stripped_dag.remove_op_node(vertex)
    return stripped_dag

def generate_random_circuit(num_qubits, circuit_depth, density, inverse):
    circuit = QuantumCircuit(num_qubits,name='q')
    max_gates_per_layer = int(num_qubits/2)
    num_gates_per_layer = max(int(density*max_gates_per_layer),1)
    # print('Generating %d-q random circuit, density = %d*%d.'%(
    #     num_qubits,num_gates_per_layer,circuit_depth))
    depth_of_random = int(circuit_depth/4) if inverse else int(circuit_depth/2)
    for depth in range(depth_of_random):
        qubit_candidates = list(range(num_qubits))
        num_gates = 0
        while len(qubit_candidates)>=2 and num_gates<num_gates_per_layer:
            qubit_pair = np.random.choice(a=qubit_candidates,replace=False,size=2)
            for qubit in qubit_pair:
                del qubit_candidates[qubit_candidates.index(qubit)]
            # Add a 2-qubit gate
            qubit_pair = [circuit.qubits[qubit] for qubit in qubit_pair]
            circuit.append(instruction=CXGate(),qargs=qubit_pair)
            num_gates += 1
        # Add some 1-qubit gates
        for qubit in range(num_qubits):
            single_qubit_gate = random.choice([IGate(), RZGate(phi=random.uniform(0,np.pi*2)), SXGate(), XGate()])
            circuit.append(instruction=single_qubit_gate,qargs=[qubit])
    if inverse:
        circuit += circuit.inverse()
        solution_state = np.random.choice(2**num_qubits)
        bin_solution_state = bin(solution_state)[2:].zfill(num_qubits)
        bin_solution_state = bin_solution_state[::-1]
        for qubit_idx, digit in zip(range(num_qubits),bin_solution_state):
            if digit=='1':
                circuit.append(instruction=XGate(),qargs=[circuit.qubits[qubit_idx]])
    return circuit