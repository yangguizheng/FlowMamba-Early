"""Run the jobs of OUTPUT_ROOT/experiments/jobs.jsonl one after another.

A job is complete when its result.json exists. Before starting a job the worker creates
experiments/claims/<run_id> with O_EXCL, so several workers can share one job file. The job
file is read again before every job, so jobs can be added or reordered while workers run.
Creating experiments/STOP_<worker name> makes that worker exit after its current job. A
failed job keeps its log and its claim, is listed in experiments/failed_jobs.txt and is not
retried automatically.

    python src/worker.py w1
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import experiment

OUT = experiment.OUT
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def log(name, msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{name}] {msg}"
    print(line, flush=True)
    with open(f"{OUT}/queue.log", "a") as f:
        f.write(line + "\n")


def main():
    name = sys.argv[1]
    os.makedirs(f"{OUT}/claims", exist_ok=True)
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    while True:
        if os.path.exists(f"{OUT}/STOP_{name}"):
            log(name, "stop file found, exiting")
            return
        job = tag = run_dir = None
        for cand in (json.loads(line) for line in open(f"{OUT}/jobs.jsonl") if line.strip()):
            run_dir, tag = experiment.run_dir_for(cand)
            if os.path.exists(f"{run_dir}/result.json"):
                continue
            try:
                fd = os.open(f"{OUT}/claims/{tag}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, name.encode())
                os.close(fd)
                job = cand
                break
            except FileExistsError:
                continue
        if job is None:
            break
        log(name, f"start {tag}")
        t0 = time.time()
        with open(f"{OUT}/logs/{tag}.log", "w") as out:
            rc = subprocess.run([PY, f"{HERE}/train_job.py", json.dumps(job)], stdout=out, stderr=subprocess.STDOUT, cwd=HERE).returncode
        status = "ok" if rc == 0 and os.path.exists(f"{run_dir}/result.json") else f"FAILED rc={rc}"
        log(name, f"end {tag} {status} {(time.time() - t0) / 60:.1f} min")
        if status != "ok":
            with open(f"{OUT}/failed_jobs.txt", "a") as f:
                f.write(f"{tag}\trc={rc}\tsee logs/{tag}.log\n")
    log(name, "queue finished")
    open(f"{OUT}/WORKER_{name}_DONE", "w").write(datetime.now().isoformat())


if __name__ == "__main__":
    main()
