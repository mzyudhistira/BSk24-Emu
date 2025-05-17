from pathlib import Path


def initialize_directory():
    """
    Initialize data directories
    """
    input_dir = Path("data/input")
    model_dir = Path("data/model")
    training_dir = Path("data/training")
    output_dir = Path("data/output")

    input_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def initialize_run_result_dir(run_param, run_datetime):
    """
    Initialize the directory of the run result

    Args:
        run_param (dict): The dict from the run config
        run_datetime (str): Datetime of the run

    Returns:
        run_dir (str): Run result directory
    """
    run_dir_name = f"{run_datetime}_{run_param['run']['name']}"
    run_dir = Path(f"data/result/{run_dir_name}")
    run_dir.mkdir(parents=True, exist_ok=True)  # change exist_ok after done

    return run_dir
