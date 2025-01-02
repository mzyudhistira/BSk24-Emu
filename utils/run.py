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
    with open(config_file, 'r') as f:
        return json.load(f)

def parse_args(): 
    """Add command line argument for the run

        argparse.Namespace: parse argument
    """    
    parser = argparse.ArgumentParser(description='Run the program with additional arguments')
    # parser.add_argument('--config', type=str, help='Path to the parameters configuration (JSON)', required=True)
    parser.add_argument('--parallel', action='store_true', help='Run the program in parallel')

    return parser.parse_args()
