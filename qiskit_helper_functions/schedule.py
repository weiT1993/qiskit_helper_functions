import math, copy, random, pickle, logging
import numpy as np
from qiskit.compiler import transpile, assemble
from qiskit import Aer, execute, QuantumCircuit
from typing import Dict, Any
from time import time
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, RuntimeJob
from qiskit_ibm_provider import least_busy, IBMBackend

from qiskit_helper_functions.non_ibmq_functions import apply_measurement
from qiskit_helper_functions.ibmq_functions import get_backend_info
from qiskit_helper_functions.conversions import dict_to_array, memory_to_dict


class JobItem:
    def __init__(self, backend: IBMBackend) -> None:
        """
        One IBM job.
        max_circuits: max number of circuits allowed in a job
        max_shots: max number of shots per circuit allowed in a job
        """
        self.backend = backend
        self.max_circuits = backend.max_circuits
        self.max_shots = backend.max_shots
        self.job_shots = 0
        self.circuit_names = []
        self.circuits = []

    def add_circuit(
        self, circuit_name: str, circuit: QuantumCircuit, shots: int
    ) -> int:
        """
        Add a circuit to the job.
        Return the remaining shots for the circuit not added into the current job.
        """
        num_vacant_slots = self.max_circuits - len(self.circuits)
        if num_vacant_slots > 0:
            circuit_shots = max(shots, self.job_shots)
            circuit_shots = min(circuit_shots, self.max_shots)
            num_slots = math.ceil(shots / circuit_shots)
            slots_to_add = min(num_slots, num_vacant_slots)
            self.circuits += [circuit] * slots_to_add
            self.circuit_names += [circuit_name] * slots_to_add
            self.job_shots = circuit_shots
            remaining_shots = shots - slots_to_add * self.job_shots
        else:
            remaining_shots = shots
        return remaining_shots

    def is_full(self) -> bool:
        """
        Return True if the job cannot hold more circuits
        """
        return len(self.circuits) == self.max_circuits

    def submit(self) -> None:
        """
        Submit the job and save job ID
        """
        sampler = Sampler(backend=self.backend)  # type: ignore
        job = sampler.run(circuits=self.circuits, shots=self.job_shots)
        logging.info(
            "Submit job {} to {:s} --> {:d} distinct circuits. {:d} total circuits. {:d} shots each.".format(
                job.job_id(),
                self.backend.name,
                len(set(self.circuit_names)),
                len(self.circuit_names),
                self.job_shots,
            )
        )
        self.job_id = job.job_id()


class Scheduler:
    """
    IBM job submission/simulating backend
    Compile as many circuits as possible into a single IBM job
    """

    def __init__(self, circuits: Dict[str, Any]):
        """
        Input:
        circuits[circuit_name]: circuit, shots
        """
        self.circuits = circuits
        for circuit_name in circuits:
            for field in ["circuit", "shots"]:
                assert field in circuits[circuit_name], "{:s} misses {:s}".format(
                    circuit_name, field
                )
            if circuits[circuit_name]["circuit"].num_clbits == 0:
                circuits[circuit_name]["circuit"].measure_all()

    def add_ibm_account(self, token: str, instance: str):
        """
        Have to run this function first before submitting jobs to IBMQ or using noisy simulations

        from qiskit_ibm_provider import IBMProvider
        # Save your credentials on disk.
        # IBMProvider.save_account(token='<IBM Quantum API key>')
        provider = IBMProvider(instance='ibm-q-bnl/c2qa-projects/tang-quantum-dev')
        """
        self.token = token
        self.instance = instance
        QiskitRuntimeService.save_account(
            channel="ibm_quantum", token=self.token, overwrite=True
        )

    def submit_ibm_jobs(self, backend_selection_mode: str, real_device: bool) -> None:
        """
        Submit the circuits to IBM devices.
        Two available device selection modes:
        - "least_busy": choose the least busy devices
        - "best": choose the highest average fidelity devices

        transpilation: whether to transpile the circuits or run as is
        real_device: whether to run on real device or simulation
        """
        logging.info(
            "--> IBM Scheduler : Submitting Jobs <--",
        )
        service = QiskitRuntimeService()
        if real_device:
            backends = service.backends(simulator=False, operational=True)
        else:
            backends = [service.backend("ibmq_qasm_simulator")]
        backend_info = get_backend_info(backends)  # type: ignore
        if backend_selection_mode == "least_busy":
            selection_function = lambda backend: backend_info[backend][
                "num_pending_jobs"
            ]
        elif backend_selection_mode == "best":
            selection_function = lambda backend: backend_info[backend][
                "average_gate_error"
            ]
        else:
            raise ValueError(
                "Illegal backend_selection_mode {}".format(backend_selection_mode)
            )
        jobs = {}
        for circuit_name in self.circuits:
            backend_candidates = list(
                filter(
                    lambda backend: backend_info[backend]["num_qubits"]
                    >= self.circuits[circuit_name]["circuit"].num_qubits,
                    backend_info.keys(),
                )
            )
            remaining_shots = self.circuits[circuit_name]["shots"]
            while remaining_shots > 0:
                backend = min(backend_candidates, key=selection_function)
                if backend.name not in jobs:
                    jobs[backend.name] = []
                if len(jobs[backend.name]) == 0 or jobs[backend.name][-1].is_full():
                    jobs[backend.name].append(JobItem(backend))
                remaining_shots = jobs[backend.name][-1].add_circuit(
                    circuit_name=circuit_name,
                    circuit=self.circuits[circuit_name]["circuit"],
                    shots=remaining_shots,
                )
                if jobs[backend.name][-1].is_full():
                    backend_info[backend]["num_pending_jobs"] += 1
        for backend_name in jobs:
            for job_item in jobs[backend_name]:
                job_item.submit()
        self.jobs = jobs

    def retrieve_jobs(self, force_prob, save_memory, save_directory):
        for device_name in self.device_names:
            if self.verbose:
                print("-->", "IBMQ Scheduler : Retrieving %s Jobs" % device_name, "<--")
            jobs = self.jobs[device_name]
            assert len(self.ibmq_schedules[device_name]) == len(jobs)
            memories = {}
            for job_idx in range(len(jobs)):
                schedule_item = self.ibmq_schedules[device_name][job_idx]
                hw_job = jobs[job_idx]
                if self.verbose:
                    print(
                        "Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots".format(
                            job_idx + 1,
                            len(jobs),
                            hw_job.job_id(),
                            len(schedule_item.circ_list),
                            schedule_item.total_circs,
                            schedule_item.shots,
                        ),
                        flush=True,
                    )
                ibmq_result = hw_job.result()
                start_idx = 0
                for element_ctr, element in enumerate(schedule_item.circ_list):
                    key = element["key"]
                    circ = element["circ"]
                    reps = element["reps"]
                    end_idx = start_idx + reps
                    # print('{:d}: getting {:d}-{:d}/{:d} circuits, key {} : {:d} qubit'.format(element_ctr,start_idx,end_idx-1,schedule_item.total_circs-1,key,len(circ.qubits)),flush=True)
                    for result_idx in range(start_idx, end_idx):
                        ibmq_memory = ibmq_result.get_memory(result_idx)
                        if key in memories:
                            memories[key] += ibmq_memory
                        else:
                            memories[key] = ibmq_memory
                    start_idx = end_idx

            process_begin = time()
            counter = 0
            log_counter = 0
            for key in self.circ_dict:
                iteration_begin = time()
                full_circ = self.circ_dict[key]["circuit"]
                shots = self.circ_dict[key]["shots"]
                ibmq_memory = memories[key][:shots]
                mem_dict = memory_to_dict(memory=ibmq_memory)
                hw_prob = dict_to_array(
                    distribution_dict=mem_dict, force_prob=force_prob
                )
                self.circ_dict[key]["%s|hw" % device_name] = copy.deepcopy(hw_prob)
                if save_memory:
                    self.circ_dict[key]["%s_memory" % device_name] = copy.deepcopy(
                        ibmq_memory
                    )
                # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
                # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(mem_dict.values())),flush=True)
                if len(full_circ.clbits) > 0:
                    assert len(self.circ_dict[key]["%s|hw" % device_name]) == 2 ** len(
                        full_circ.clbits
                    )
                else:
                    assert len(self.circ_dict[key]["%s|hw" % device_name]) == 2 ** len(
                        full_circ.qubits
                    )
                if save_directory is not None:
                    pickle.dump(
                        self.circ_dict[key],
                        open("%s/%s.pckl" % (save_directory, key), "wb"),
                    )
                counter += 1
                log_counter += time() - iteration_begin
                if log_counter > 60 and self.verbose:
                    elapsed = time() - process_begin
                    eta = elapsed / counter * len(self.circ_dict) - elapsed
                    print(
                        "Processed %d/%d circuits, elapsed = %.3e, ETA = %.3e"
                        % (counter, len(self.circ_dict), elapsed, eta),
                        flush=True,
                    )
                    log_counter = 0
