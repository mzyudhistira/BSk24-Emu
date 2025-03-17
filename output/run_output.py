import importlib

from input import run_input
from model import run_model
from generate import mass_table


def main(input):
    input_tensor = input["input"]["data"]
    trained_model = model.run_model(input["model"])
    trained_model.load_weights(input["model"]["model_weights"])

    # generate mass table
    # analyse mass table

    return [rms_deviation, std_difference, plot_deviation, plot_uncertainty, plot_loss]


if __name__ == "__main__":
    input_file = command(sys.argv)
    main(input_file)
