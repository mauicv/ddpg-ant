"""Noise schema Experiment script.

runs reinforcment learning script N for each noise schema.
"""
import os

NUMBER_RUNS      = 25
NOISE_TYPES      = ["lsn", "snn", "ou", "n"]
EPISODES_PER_RUN = 300
STEPS_PER_RUN    = 300


for noise_type in NOISE_TYPES:
    for run_no in range(NUMBER_RUNS):
        opts = f"--dir={noise_type}_{run_no} -s {STEPS_PER_RUN} -e {EPISODES_PER_RUN} -nt {noise_type}"
        opts = 'train '
        for opt in [f" -d '{noise_type}_{run_no}'",
                    f" -s {STEPS_PER_RUN}",
                    f" -e {EPISODES_PER_RUN}",
                    f" -nt {noise_type}"]:
            opts += opt
        os.system(f"python main.py {opts}")
