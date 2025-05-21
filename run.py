import os
import multiprocessing
from datetime import datetime
import csv
import shutil

import numpy as np
import pandas as pd

import input
import model
import training
import output
import utils
import analysis


def run(config):
    """

    Args:
        config (str): Location of the TOML run configuration file

    Returns:
        run_summary (list): List of strings summarizing the run:
                            datetime, run name, run config file, loss, validation loss,
                            rms deviation, mae,std difference, output file, run time and note

    """
    # Load run config
    run_param = utils.run.load_param(config)
    start_time = datetime.now()
    run_datetime = start_time.strftime("%Y-%m-%d %H:%M")
    cache_folder = Path("data/cache")
    cache_folder.mkdir(parents=True, exist_ok=True)

    # Initialize the directories
    utils.file.initialize_directory()
    run_dir = utils.file.initialize_run_result_dir(run_param, run_datetime)
    run_param["run"]["dir"] = run_dir

    # Running the pipeline
    input_obj = input.build.Input(run_param)

    model_obj = model.build.Model(input_obj, run_param)
    model_obj.model.summary()

    train_obj = training.train.TrainModel(run_param)
    train_obj.run_training(input_obj, model_obj)

    test_result = output.test.Test(input_obj, model_obj, train_obj, run_param)
    analysis.plot.plot_loss(train_obj.loss, train_obj.val_loss)

    # Writing run summary
    end_time = datetime.now()
    run_time = end_time - start_time

    run_summary = [
        run_datetime,
        run_param["run"]["name"],
        config,
        train_obj.loss,
        train_obj.val_loss,
        test_result.rms_dev,
        test_result.mae,
        test_result.std_diff,
        test_result.output,
        run_time,
        run_param["run"]["note"],
    ]

    with open("summary.csv", "a", newline="") as summary_file:
        writer = csv.writer(summary_file)

        run_summary = [
            [x if x is not None else "" for x in run_summary]
        ]  # Replace None with empty string
        writer.writerows(run_summary)

    shutil.rmtree(cache_folder)

    return run_summary


def main():
    file, parallel = utils.run.parse_args()
    file_extension = os.path.splitext(file)[1]
    multiple_run_configs = file_extension == ".txt"

    if multiple_run_configs and parallel:
        print("Running in parallel")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            run_configs = [config for config in utils.run.load_run_config_files(file)]
            pool.map(run, run_configs)

    elif multiple_run_configs:
        print("Running in sequential")
        run_configs = [config for config in utils.run.load_run_config_files(file)]
        for run_config in run_configs:
            run(run_config)

    else:
        print("Running")
        run(file)


if __name__ == "__main__":
    main()
