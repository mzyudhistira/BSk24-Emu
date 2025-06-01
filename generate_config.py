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
percentage = np.linspace(0.1, 0.9, 9)

for i in percentage:
    config = utils.run.load_param(base_config)
    config["run"]["name"] = f"Dataset={i*100:.1f}%"
    config["run"]["note"] = float(i)
    config["input"]["param"]["dataset"]["variant_percentage"] = float(i)
    config_name = f"data/run/{run_dir_name}/config_{i:.2f}.toml"

    with open(config_name, "w") as f:
        toml.dump(config, f)

    with open(multiple_config, "a") as f:
        f.write(f"{config_name}\n")
