"""
Job submission/simulating backend
Input:
circ_dict (dict): circ (not transpiled), shots, evaluator_info (optional)
"""

import math, copy, random, pickle
import numpy as np
from qiskit.compiler import transpile, assemble
from qiskit import Aer
from time import time

from qiskit_helper_functions.non_ibmq_functions import apply_measurement
from qiskit_helper_functions.ibmq_functions import get_device_info
from qiskit_helper_functions.conversions import dict_to_array, memory_to_dict

class ScheduleItem:
    def __init__(self,max_experiments,max_shots):
        self.max_experiments = max_experiments
        self.max_shots = max_shots
        self.circ_list = []
        self.shots = 0
        self.total_circs = 0
    
    def update(self, key, circ, shots):
        reps_vacant = self.max_experiments - self.total_circs
        if reps_vacant>0:
            circ_shots = max(shots,self.shots)
            circ_shots = min(circ_shots,self.max_shots)
            total_reps = math.ceil(shots/circ_shots)
            reps_to_add = min(total_reps,reps_vacant)
            circ_list_item = {'key':key,'circ':circ,'reps':reps_to_add}
            self.circ_list.append(circ_list_item)
            self.shots = circ_shots
            self.total_circs += reps_to_add
            shots_remaining = shots - reps_to_add * self.shots
        else:
            shots_remaining = shots
        return shots_remaining

class Scheduler:
    def __init__(self,circ_dict,token,hub,group,project,device_name,datetime):
        self.circ_dict = circ_dict
        self.token = token
        self.hub = hub
        self.group = group
        self.project = project
        self.device_name = device_name
        self.datetime = datetime
        self.device_info = get_device_info(token=self.token,hub=self.hub,group=self.group,project=self.project,device_name=self.device_name,
        fields=['device','properties','basis_gates','coupling_map','noise_model'],datetime=self.datetime)
        self.device_size = len(self.device_info['properties'].qubits)
        self.check_input()
        self.schedule = self.get_schedule(device_max_shots=self.device_info['device'].configuration().max_shots,
        device_max_experiments=self.device_info['device'].configuration().max_experiments)

    def check_input(self):
        assert isinstance(self.circ_dict,dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if 'circuit' not in value or 'shots' not in value:
                raise Exception('Input circ_dict should have `circuit`, `shots` for key {}'.format(key))
            elif value['circuit'].num_qubits > self.device_size:
                raise Exception('Input `circuit` for key {} has {:d}-q ({:d}-q device)'.format(key,value['circuit'].num_qubits,self.device_size))

    def get_schedule(self,device_max_shots,device_max_experiments):
        circ_dict = copy.deepcopy(self.circ_dict)
        schedule = []
        schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
        key_idx = 0
        while key_idx<len(circ_dict):
            key = list(circ_dict.keys())[key_idx]
            circ = circ_dict[key]['circuit']
            shots = circ_dict[key]['shots']
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            shots_remaining = schedule_item.update(key,circ,shots)
            if shots_remaining>0:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(schedule_item)
                schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
                circ_dict[key]['shots'] = shots_remaining
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                circ_dict[key]['shots'] = shots_remaining
                key_idx += 1
        if schedule_item.total_circs>0:
            schedule.append(schedule_item)
        return schedule

    def submit_jobs(self,real_device,transpilation,verbose=False):
        if verbose:
            print('*'*20,'Submitting jobs','*'*20,flush=True)
        self.jobs = []
        for idx, schedule_item in enumerate(self.schedule):
            # print('Submitting job %d/%d'%(idx+1,len(schedule)))
            # print('Has %d total circuits * %d shots, %d circ_list elements'%(schedule_item.total_circs,schedule_item.shots,len(schedule_item.circ_list)))
            job_circuits = []
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                # print('Key {}, {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))

                if transpilation:
                    qc=apply_measurement(circuit=circ,qubits=circ.qubits)
                    # mapped_circuit = transpile(qc,backend=self.device_info['device'],layout_method='noise_adaptive')
                    mapped_circuit = transpile(qc,backend=self.device_info['device'],optimization_level=3)
                else:
                    mapped_circuit = circ
                
                self.circ_dict[key]['mapped_circuit'] = mapped_circuit

                # print('scheduler:')
                # print(mapped_circuit)
                circs_to_add = [mapped_circuit]*reps
                job_circuits += circs_to_add
            
            assert len(job_circuits) == schedule_item.total_circs
            
            if real_device:
                qobj = assemble(job_circuits, backend=self.device_info['device'], shots=schedule_item.shots,memory=True)
                hw_job = self.device_info['device'].run(qobj)
            else:
                qobj = assemble(job_circuits, backend=Aer.get_backend('qasm_simulator'), shots=schedule_item.shots, memory=True)
                # hw_job = Aer.get_backend('qasm_simulator').run(qobj,noise_model=self.device_info['noise_model])
                hw_job = Aer.get_backend('qasm_simulator').run(qobj)
            self.jobs.append(hw_job)
            if verbose:
                print('Submitting job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(idx+1,len(self.schedule),hw_job.job_id(),len(schedule_item.circ_list),len(job_circuits),schedule_item.shots),flush=True)

    def retrieve_jobs(self,force_prob,save_memory,save_directory,verbose=False):
        if verbose:
            print('*'*20,'Retrieving jobs','*'*20)
        assert len(self.schedule) == len(self.jobs)
        memories = {}
        for job_idx in range(len(self.jobs)):
            schedule_item = self.schedule[job_idx]
            hw_job = self.jobs[job_idx]
            if verbose:
                print('Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(
                    job_idx+1,len(self.jobs),hw_job.job_id(),
                    len(schedule_item.circ_list),schedule_item.total_circs,schedule_item.shots),flush=True)
            hw_result = hw_job.result()
            start_idx = 0
            for element_ctr, element in enumerate(schedule_item.circ_list):
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                end_idx = start_idx + reps
                # print('{:d}: getting {:d}-{:d}/{:d} circuits, key {} : {:d} qubit'.format(element_ctr,start_idx,end_idx-1,schedule_item.total_circs-1,key,len(circ.qubits)),flush=True)
                for result_idx in range(start_idx,end_idx):
                    experiment_hw_memory = hw_result.get_memory(result_idx)
                    if key in memories:
                        memories[key] += experiment_hw_memory
                    else:
                        memories[key] = experiment_hw_memory
                start_idx = end_idx
        process_begin = time()
        counter = 0
        log_frequency = int(len(self.circ_dict)/5) if len(self.circ_dict)>5 else 1
        for key in self.circ_dict:
            full_circ = self.circ_dict[key]['circuit']
            shots = self.circ_dict[key]['shots']
            memory = memories[key][:shots]
            mem_dict = memory_to_dict(memory=memory)
            hw_prob = dict_to_array(distribution_dict=mem_dict,force_prob=force_prob)
            self.circ_dict[key]['prob'] = copy.deepcopy(hw_prob)
            if save_memory:
                self.circ_dict[key]['memory'] = copy.deepcopy(memory)
            # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
            # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(mem_dict.values())),flush=True)
            if len(full_circ.clbits)>0:
                assert len(self.circ_dict[key]['prob']) == 2**len(full_circ.clbits)
            else:
                assert len(self.circ_dict[key]['prob']) == 2**len(full_circ.qubits)
            if save_directory is not None:
                pickle.dump(self.circ_dict[key], open('%s/%s.pckl'%(save_directory,key),'wb'))
            counter += 1
            elapsed = time() - process_begin
            eta = elapsed/counter*len(self.circ_dict)-elapsed
            if verbose and counter%log_frequency==0:
                print('Processed %d/%d circuits, elapsed = %.3e, ETA = %.3e'%(counter,len(self.circ_dict),elapsed,eta))