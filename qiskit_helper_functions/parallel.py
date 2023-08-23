from typing import List, Iterable
import numpy as np
from functools import partial
import multiprocessing, os, math


def find_process_jobs(jobs, rank, num_workers):
    count = int(len(jobs) / num_workers)
    remainder = len(jobs) % num_workers
    if rank < remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs


def make_job_chunks(num_jobs: int) -> List[List[np.ndarray]]:
    """
    Group the number of jobs into chunks for starmap processing
    """
    num_chunks = min(100, num_jobs)
    chunks = []
    for chunk_index in range(num_chunks):
        chunk_job_indices = find_process_jobs(range(num_jobs), chunk_index, num_chunks)
        chunk_job_indices = np.array(chunk_job_indices)
        chunks.append([chunk_job_indices])
    return chunks


def pool_process(partial_fun: partial, num_workers: int | None, inputs: List):
    """
    Run the partial_fun with the given inputs in parallel
    """
    chunksize = math.ceil(len(inputs) / 100)
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.starmap(partial_fun, inputs, chunksize=chunksize)
    pool.close()
    return results
