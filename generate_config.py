import toml
from datetime import datetime
from pathlib import Path

import numpy as np

import utils


"""
Modify this file based on the need
"""

base_config = "data/run/test.toml"
run_dir_name = datetime.now().strftime("%Y-%m-%d %H:%M")
run_dir = Path(f"data/run/{run_dir_name}")
run_dir.mkdir(parents=True, exist_ok=True)

multiple_config = run_dir / "mult.txt"

for i in range(1, 8):
    config = utils.run.load_param(base_config)
    config["run"]["name"] = f"Linear Neurons Fix Diff: {i}"
    config["run"]["note"] = int(i)
    config["input"]["param"]["dataset"]["variant_percentage"] = 0.8
    neurons = [16 * i, 16 * (i + 1), 16 * (i + 2)]

    config["model"]["param"]["neurons"] = neurons

    config_name = f"data/run/{run_dir_name}/config_{i}.toml"

    with open(config_name, "w") as f:
        toml.dump(config, f)

    with open(multiple_config, "a") as f:
        f.write(f"{config_name}\n")
