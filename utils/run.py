import os
import argparse
import multiprocessing

import tomllib


def load_param(run_config_file):
    """

    Args:
        run_config_file (str): File path of the run config (in toml)

    Returns:
        config (dict) : configuration of the run

    """
    with open(run_config_file, "rb") as f:
        config = tomllib.load(f)

    return config


def load_run_config_files(run_config_files):
    """
    Read all run config files from TXT input

    Args:
        run_config_files (str): File path of the list of run config files (in txt)

    Returns:
        config_files (list): list of configuration files' path

    """
    with open(run_config_files, "r") as f:
        config_files = [line.strip() for line in f if line.strip()]

    return config_files


def parse_args():
    """
    Read the given shell arguments

    Returns:
        args.file (str) : run config files
        args.parallel (bool): True if the user puts parallel flag

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", help="Run configuration file in TOML or TXT (see documentation)"
    )
    parser.add_argument(
        "-pl",
        "--parallel",
        help="Run the program in parallel, only accepts TXT",
        action="store_true",
    )

    args = parser.parse_args()

    if args.parallel and not args.file.lower().endswith(".txt"):
        parser.error("Parallel mode only supports TXT files.")

    return args.file, args.parallel
