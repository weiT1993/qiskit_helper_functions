import os
import pickle
import math
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.classicalregister import ClassicalRegister
import qiskit.circuit.library as library
import numpy as np

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore, gen_adder, gen_grover
from helper_functions.conversions import dict_to_array

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

def apply_measurement(circuit):
    c = ClassicalRegister(len(circuit.qubits), 'c')
    meas = QuantumCircuit(circuit.qregs[0], c)
    meas.barrier(circuit.qubits)
    meas.measure(circuit.qubits,c)
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
        full_circ = gen_hwea(i*j,1)
    elif circuit_type == 'bv':
        full_circ = gen_BV(gen_secret(i*j),barriers=False)
    elif circuit_type == 'qft':
        full_circ = library.QFT(num_qubits=full_circ_size,approximation_degree=0,do_swaps=False)
    elif circuit_type=='aqft':
        approximation_degree=int(math.log(full_circ_size,2)+2)
        full_circ = library.QFT(num_qubits=full_circ_size,approximation_degree=full_circ_size-approximation_degree,do_swaps=False)
    elif circuit_type == 'sycamore':
        full_circ = gen_sycamore(i,j,8)
    elif circuit_type == 'adder':
        if full_circ_size%2==0:
            full_circ = gen_adder(nbits=int((full_circ_size-2)/2),barriers=False,regname='q')
        else:
            full_circ = QuantumCircuit()
    elif circuit_type == 'grover':
        if full_circ_size%2==1:
            full_circ = gen_grover(width=int((full_circ_size+1)/2))
        else:
            full_circ = QuantumCircuit()
    else:
        raise Exception('Illegal circuit type:',circuit_type)
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
        backend_options = {'max_memory_mb': 2**30*16/1024**2}
        if isinstance(options,dict) and 'num_shots' in options:
            num_shots = options['num_shots']
        else:
            num_shots = max(1024,2**circuit.num_qubits)
        backend = Aer.get_backend('qasm_simulator')
        qc = apply_measurement(circuit=circuit)

        if isinstance(options,dict) and 'memory' in options:
            memory = options['memory']
        else:
            memory = False
        noiseless_qasm_result = execute(qc, backend, shots=num_shots, backend_options=backend_options, memory=memory).result()

        if memory:
            qasm_memory = np.array(noiseless_qasm_result.get_memory(0))
            assert len(qasm_memory)==num_shots
            return qasm_memory
        else:
            noiseless_counts = noiseless_qasm_result.get_counts(0)
            assert sum(noiseless_counts.values())==num_shots
            noiseless_counts = dict_to_array(distribution_dict=noiseless_counts,force_prob=True)
            return noiseless_counts
    else:
        raise Exception('Backend %s illegal'%backend)