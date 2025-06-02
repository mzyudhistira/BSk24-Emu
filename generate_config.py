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
percentage = np.linspace(0.1, 1, 10)
variant_id = random.randint(1, 11023)

for i in percentage:
    config = utils.run.load_param(base_config)
    iw = float(round(i, 1))

    config["run"]["name"] = f"Variant:{variant_id}, Dataset:{iw*100*0.8}%"
    config["run"]["note"] = float(iw * 100 * 0.8)
    config["input"]["param"]["dataset"]["variant_percentage"] = iw
    config["input"]["param"]["dataset"]["variant_id"] = variant_id

    config_name = f"data/run/{run_dir_name}/config_{iw}.toml"

    with open(config_name, "w") as f:
        toml.dump(config, f)

    with open(multiple_config, "a") as f:
        f.write(f"{config_name}\n")
