import sys
import os
import subprocess
import tempfile
import json
import csv
from datetime import datetime


def run_module(module, input_data):
    """_summary_

    Args:
        module (_type_): _description_
        input_data (_type_): _description_
    """
    # Create a temporary file to store the input data as JSON
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        temp_file_path = temp_file.name
        json.dump(input_data, temp_file)

    try:
        # Run the module's run.py script using the given input
        module_run_script = os.path.join(module, f"run_{module}.py")
        result = subprocess.run(
            ["python", module_run_script, temp_file_path],
            capture_output=True,
            text=True,
        )

    finally:
        # Clean up the temporary JSON file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def run(input_file):
    """Running a machine learning pipeline from the thesis

    Args:
        input_file (string): Path of the JSON input file
    """
    if not os.path.isfile(input_file):
        print(f"Input file not found at {input_file}")
        sys.exit(1)

    with open(input_file, "r") as file:
        run_data = json.load(file)

    start_time = datetime.now()
    run_name = run_data["run"]["name"]
    run_note = run_data["run"]["note"]
    run_date = start_time.strftime("%Y-%m-%d %H:%M:%S")
    bsk24 = run_data["input"]["BSk24"]
    bsk24_variant = run_data["input"]["BSk24_variant"]

    print(40 * "-" + "\nInitializing input data\n" + 40 * "-")
    input_data = run_module("input", run_data["input"])

    print(40 * "-" + "\nInitializing model\n" + 40 * "-")
    model = run_module("model", run_data["model"])

    print(40 * "-" + "\nTraining model\n" + 40 * "-")
    model_input = {
        "input": {"data": input_data},
        "model": {"model": model},
        "training": run_data["training"],
    }
    model_weights, loss, val_loss = run_module("training", model_input)
    run_data["model"]["model_weights"] = model_weights
    run_data["training"].update({"loss": loss, "val_loss": val_loss})

    print(40 * "-" + "\nPredicting output\n" + 40 * "-")
    prediction_input = {
        "input": {"data": input_data},
        "model": run_data["model"],
        "training": run_data["training"],
    }
    result = run_module("output", prediction_input)
    run_summary = [run_name, run_date, bsk24, bsk24_variant] + result + [run_note]

    end_time = datetime.now()
    run_duration = end_time - start_time

    print(40 * "-" + "\nSaving results\n" + 40 * "-")
    with open("data/analysis/result.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(run_summary)

    print("\n\n")
    print(40 * "-")
    print(f"\nrun_duration: {run_duration.total_seconds()} s")
    print(f"\nrms_deviation: {result[0]}")
    print(f"\nstd_difference: {result[1]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input file!")
        sys.exit()

    input_file = sys.argv[1]
    run(input_file)
