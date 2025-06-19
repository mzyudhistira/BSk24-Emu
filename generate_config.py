import toml
from datetime import datetime
from pathlib import Path
import random

import numpy as np

import utils


"""
Modify this file based on the need
"""

base_config = "data/run/test.toml"
run_dir_name = datetime.now().strftime("%Y-%m-%d %H:%M")
run_dir = Path(f"data/run/{run_dir_name}")
run_dir.mkdir(parents=True, exist_ok=True)

percentage = np.linspace(0.05, 1, 15)

with open("variants_test.txt", "r") as file:
    lines = file.readlines()

variants_id = list(range(1, 11023))
variant_run_idx = 1

for i in range(int(np.ceil(11022 / 15))):
    multiple_config = run_dir / f"mult_{i}.txt"

    for j in range(15):
        config = utils.run.load_param(base_config)

        config["run"]["name"] = f"variant_{variant_run_idx}"
        config["input"]["param"]["dataset"]["variant_percentage"] = 0.3
        config["input"]["param"]["dataset"]["variant_id"] = variant_run_idx

        config_name = f"data/run/{run_dir_name}/variant_{variant_run_idx}.toml"

        with open(config_name, "w") as f:
            toml.dump(config, f)

        with open(multiple_config, "a") as f:
            f.write(f"{config_name}\n")

        variant_run_idx += 1

    with open(run_dir / "run.sh", "a") as f:
        f.write(f"python run.py {multiple_config} -pl\n")
