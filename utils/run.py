import sys
import argparse
import json
import multiprocessing
import os
import numpy as np
from typing import List, Dict


def load_param(param_json):
    """
    Load parameters used for the training
    """
    with open(config_file, "r") as f:
        return json.load(f)


def parse_args():
    """Add command line argument for the run

    argparse.Namespace: parse argument
    """
    parser = argparse.ArgumentParser(
        description="Run the program with additional arguments"
    )
    # parser.add_argument('--config', type=str, help='Path to the parameters configuration (JSON)', required=True)
    parser.add_argument(
        "--parallel", action="store_true", help="Run the program in parallel"
    )

    return parser.parse_args()


def command(arg):
    """Get the command line input and parse the input file

    Args:
        arg (sys.argv): command line input

    Returns:
        input_file (string): directory of the input file stated in the command line input
    """
    if len(arg) < 2:
        print("Please provide an input file!")
        sys.exit()

    input_file = arg[1]

    return input_file
