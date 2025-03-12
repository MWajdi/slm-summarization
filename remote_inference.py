import paramiko
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

USERNAME = "username"
PASSWORD = "password"

# List of remote machines
REMOTE_MACHINES = [
    "acromion.polytechnique.fr",
    "apophyse.polytechnique.fr",
    "atlas.polytechnique.fr",
    "axis.polytechnique.fr",
    "coccyx.polytechnique.fr",
    "cote.polytechnique.fr",
    "cubitus.polytechnique.fr",
    "cuboide.polytechnique.fr",
    "femur.polytechnique.fr",
    "frontal.polytechnique.fr",
    "humerus.polytechnique.fr",
    "malleole.polytechnique.fr",
    "metacarpe.polytechnique.fr",
    "parietal.polytechnique.fr",
    "perone.polytechnique.fr",
    "phalange.polytechnique.fr",
    "radius.polytechnique.fr",
    "rotule.polytechnique.fr",
    "sacrum.polytechnique.fr",
    "sternum.polytechnique.fr",
    "tarse.polytechnique.fr",
    "temporal.polytechnique.fr",
    "tibia.polytechnique.fr",
    "xiphoide.polytechnique.fr"
]


complete_chunks = [0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 15, 16, 17, 18]
missing_chunks = [3, 10, 11, 12, 14, 19, 20, 21, 22, 24]
incomplete_chunks = [23]

# Each tuple: (machine hostname, chunk index)
# MACHINE_JOBS = [
#     # ("machine1.domain", 0),
#     (machine, i) 
#     for i, machine in enumerate(REMOTE_MACHINES)
# ]

MACHINE_JOBS = [
    (REMOTE_MACHINES[i % len(REMOTE_MACHINES)], chunk_idx)
    for i, chunk_idx in enumerate(missing_chunks + incomplete_chunks)
]

def run_inference_on_remote(hostname, chunk_idx):
    """SSH to a single machine, activate venv, run inference on the assigned CSV."""
    print(f"[{hostname}] Connecting...")

    # Build command
    chunk_path = f"~/NLP_project/datasets/split_{chunk_idx}.csv"
    inference_cmd = (
        f"bash -c 'source /Data/venv/bin/activate && "
        f"python ~/NLP_project/inference.py {chunk_path}'"
    )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=USERNAME, password=PASSWORD)

    print(f"[{hostname}] Running inference on chunk {chunk_idx}")
    stdin, stdout, stderr = ssh.exec_command(inference_cmd)

    out = stdout.read().decode()
    err = stderr.read().decode()

    ssh.close()

    if out.strip():
        print(f"[{hostname}] STDOUT:\n{out}")
    if err.strip():
        print(f"[{hostname}] STDERR:\n{err}")

    print(f"[{hostname}] Done with chunk {chunk_idx}")

def main():
    # Create a ThreadPoolExecutor. 
    # max_workers can be 25 (or fewer) to match your number of machines.
    with ThreadPoolExecutor(max_workers=len(MACHINE_JOBS)) as executor:
        futures = []
        # Launch each job asynchronously
        for hostname, chunk_idx in MACHINE_JOBS:
            futures.append(
                executor.submit(run_inference_on_remote, hostname, chunk_idx)
            )

        # Optionally wait for all tasks to complete (and handle results/exceptions)
        for future in as_completed(futures):
            try:
                future.result()  # will raise if there's an exception in the thread
            except Exception as e:
                print(f"Exception during SSH job: {e}")


main()