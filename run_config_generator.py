import toml
from datetime import datetime
from pathlib import Path

import utils


"""
Modify this file based on the need
"""

base_config = "data/run/test.toml"
run_dir_name = datetime.now().strftime("%Y-%m-%d %H:%M")
run_dir = Path(f"data/run/{run_dir_name}")
run_dir.mkdir(parents=True, exist_ok=True)

multiple_config = run_dir / "mult.toml"

for i in range(1, 11):
    config = utils.run.load_param(base_config)
    config["input"]["param"]["N_input"] = i
    config_name = f"data/run/{run_dir_name}/config_{i}.toml"

    with open(config_name, "w") as f:
        toml.dump(config, f)

    with open(multiple_config, "a") as f:
        f.write(f"{config_name}\n")
