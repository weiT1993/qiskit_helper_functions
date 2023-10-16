import math, copy, random, pickle, logging
import numpy as np
from qiskit import Aer, execute, QuantumCircuit
from typing import Dict, Any
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, RuntimeJob
from qiskit_ibm_provider import IBMBackend

from qiskit_helper_functions.ibmq_functions import get_backend_info
from qiskit_helper_functions.conversions import (
    quasi_dist_to_array,
)


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
        - optimization_level=3 adds dynamical decoupling
        - resilience_level=1 adds readout error mitigation
        """
        sampler = Sampler(backend=self.backend)  # type: ignore
        job = sampler.run(
            circuits=self.circuits,
            shots=self.job_shots,
            optimization_level=3,
            resilience_level=1,
        )
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

    def retrieve(self, service: QiskitRuntimeService) -> None:
        """
        Retrieve the job results
        Compile results from the same circuit names
        """
        quasi_arrays = {}
        job_result = service.job(self.job_id).result()
        for circuit_name, circuit, circuit_quasi_dist in zip(
            self.circuit_names, self.circuits, job_result.quasi_dists
        ):
            circuit_quasi_array = quasi_dist_to_array(
                quasi_dist=circuit_quasi_dist, num_qubits=circuit.num_qubits
            )
            if circuit_name in quasi_arrays:
                quasi_arrays[circuit_name] += circuit_quasi_array
            else:
                quasi_arrays[circuit_name] = circuit_quasi_array
        self.circuit_results = {}
        for circuit_name in quasi_arrays:
            self.circuit_results[circuit_name] = (
                quasi_arrays[circuit_name]
                / np.sum(quasi_arrays[circuit_name])
                * self.job_shots
            )


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
            backends = [service.backend("simulator_statevector")]
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

    def retrieve_ibm_jobs(
        self,
    ) -> None:
        """
        Retrieve IBM jobs
        """
        service = QiskitRuntimeService()
        for backend_name in self.jobs:
            for job_item in self.jobs[backend_name]:
                job_item.retrieve(service)
                for circuit_name in job_item.circuit_results:
                    if "output" not in self.circuits[circuit_name]:
                        self.circuits[circuit_name][
                            "output"
                        ] = job_item.circuit_results[circuit_name]
                    else:
                        self.circuits[circuit_name][
                            "output"
                        ] += job_item.circuit_results[circuit_name]
        for circuit_name in self.circuits:
            self.circuits[circuit_name]["output"] = self.circuits[circuit_name][
                "output"
            ] / np.sum(self.circuits[circuit_name]["output"])
