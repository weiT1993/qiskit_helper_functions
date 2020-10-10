import os
import pickle
import math
from datetime import datetime
import subprocess
from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import CouplingMap
from qiskit.circuit.classicalregister import ClassicalRegister
import qiskit.circuit.library as library

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore, gen_adder, gen_grover

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

def load_IBMQ(token,hub,group,project):
    token = token
    # token = '796046b1d3210183f406ae0849fec01e86e761dc661cfab17ba19d70ff2dbe140fc2515a0c5a88e66052122a8c0681b95ef5e3031deeb4e9284310c1c4958b56'
    if len(IBMQ.stored_account()) == 0:
        IBMQ.save_account(token)
        IBMQ.load_account()
    elif IBMQ.active_account() == None:
        IBMQ.load_account()
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    # provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy147')
    return provider

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n/val)
        if val*co_val == n:
            return val, co_val
        else:
            val -= 1

def get_device_info(device_name,fields):
    today = datetime.date(datetime.now())
    dirname = './devices/%s'%today
    filename = '%s/%s.pckl'%(dirname,device_name)
    device_info = read_dict(filename=filename)
    if len(device_info)==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        else:
            subprocess.run(['rm','-r',dirname])
            os.makedirs(dirname)
        provider = load_IBMQ()
        for x in provider.backends():
            print(x)
            if 'qasm' not in str(x):
                device = provider.get_backend(str(x))
                properties = device.properties()
                num_qubits = len(properties.qubits)
                print('Download device_info for %d-qubit %s'%(num_qubits,x))
                device = provider.get_backend(device_name)
                properties = device.properties()
                coupling_map = CouplingMap(device.configuration().coupling_map)
                noise_model = NoiseModel.from_backend(properties)
                basis_gates = noise_model.basis_gates
                device_info = {'properties':properties,
                'coupling_map':coupling_map,
                'noise_model':noise_model,
                'basis_gates':basis_gates}
                pickle.dump(device_info, open('%s/%s.pckl'%(dirname,str(x)),'wb'))
            print('-'*50)
        device_info = read_dict(filename=filename)
    for field in device_info:
        if field not in fields:
            del device_info[field]
    return device_info

def apply_measurement(circ):
    c = ClassicalRegister(len(circ.qubits), 'c')
    meas = QuantumCircuit(circ.qregs[0], c)
    meas.barrier(circ.qubits)
    meas.measure(circ.qubits,c)
    qc = circ+meas
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
        full_circ = gen_supremacy(1,full_circ_size,8)
    elif circuit_type == 'supremacy':
        if abs(i-j)<=2:
            full_circ = gen_supremacy(i,j,8)
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
        full_circ = gen_adder(nbits=int((full_circ_size-2)/2),barriers=False)
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