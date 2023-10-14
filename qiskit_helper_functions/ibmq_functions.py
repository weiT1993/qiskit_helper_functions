from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
from qiskit.transpiler import CouplingMap
from datetime import timedelta, datetime
from pytz import timezone
import time, subprocess, os, pickle, logging
from qiskit_ibm_provider import IBMBackend
from typing import List, Dict
import numpy as np

from qiskit_helper_functions.non_ibmq_functions import read_dict


def get_backend_info(backends: List[IBMBackend]) -> Dict:
    """
    Get the IBM device information:
    - Number of qubits
    - Average gate error
    - Current number of pending jobs
    """
    backend_info = {}
    for backend in backends:
        if backend.simulator:
            average_gate_error = 0
        else:
            properties = backend.properties(refresh=True).to_dict()
            backend_gate_errors = []
            for gate in properties["gates"]:
                for parameter in gate["parameters"]:
                    if parameter["name"] == "gate_error":
                        backend_gate_errors.append(parameter["value"])
            average_gate_error = np.mean(backend_gate_errors)
        backend_info[backend] = {
            "num_qubits": backend.num_qubits,
            "average_gate_error": average_gate_error,
            "num_pending_jobs": backend.status().pending_jobs,
        }
    return backend_info


def check_jobs(token, hub, group, project, cancel_jobs):
    provider = load_IBMQ(token=token, hub=hub, group=group, project=project)

    time_now = datetime.now(timezone("EST"))
    delta = timedelta(
        days=1, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0
    )
    time_delta = time_now - delta

    for x in provider.backends():
        if "qasm" not in str(x):
            device = provider.get_backend(str(x))
            properties = device.properties()
            num_qubits = len(properties.qubits)
            print(
                "%s: %d-qubit, max %d jobs * %d shots"
                % (
                    x,
                    num_qubits,
                    x.configuration().max_experiments,
                    x.configuration().max_shots,
                )
            )
            jobs_to_cancel = []
            print("QUEUED:")
            print_ctr = 0
            for job in x.jobs(limit=50, status=JobStatus["QUEUED"]):
                if print_ctr < 5:
                    print(
                        job.creation_date(),
                        job.status(),
                        job.queue_position(),
                        job.job_id(),
                        "ETA:",
                        job.queue_info().estimated_complete_time - time_now,
                    )
                jobs_to_cancel.append(job)
                print_ctr += 1
            print("RUNNING:")
            for job in x.jobs(limit=5, status=JobStatus["RUNNING"]):
                print(job.creation_date(), job.status(), job.queue_position())
                jobs_to_cancel.append(job)
            print("DONE:")
            for job in x.jobs(
                limit=5, status=JobStatus["DONE"], start_datetime=time_delta
            ):
                print(
                    job.creation_date(), job.status(), job.error_message(), job.job_id()
                )
            print("ERROR:")
            for job in x.jobs(
                limit=5, status=JobStatus["ERROR"], start_datetime=time_delta
            ):
                print(
                    job.creation_date(), job.status(), job.error_message(), job.job_id()
                )
            if cancel_jobs and len(jobs_to_cancel) > 0:
                for i in range(3):
                    print("Warning!!! Cancelling jobs! %d seconds count down" % (3 - i))
                    time.sleep(1)
                for job in jobs_to_cancel:
                    print(
                        job.creation_date(),
                        job.status(),
                        job.queue_position(),
                        job.job_id(),
                    )
                    job.cancel()
                    print("cancelled")
            print("-" * 100)
