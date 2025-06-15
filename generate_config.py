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

multiple_config = run_dir / "mult.txt"
percentage = np.linspace(0.1, 1, 5)
# variant_id = random.randint(1, 11023)
variant_id = [
    13,
    1529,
    1068,
    2498,
    2494,
    2480,
    2482,
    12,
    1,
    2484,
    2481,
    15,
    419,
    694,
    1420,
]

for i in variant_id:
    config = utils.run.load_param(base_config)

    config["run"]["name"] = f"Variant_{i}"
    config["input"]["param"]["dataset"]["variant_percentage"] = 1
    config["input"]["param"]["dataset"]["variant_id"] = i

    config_name = f"data/run/{run_dir_name}/config_{i}.toml"

    with open(config_name, "w") as f:
        toml.dump(config, f)

    with open(multiple_config, "a") as f:
        f.write(f"{config_name}\n")
