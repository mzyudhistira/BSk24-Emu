import importlib


def main(input):
    module = importlib.import_module(input["model_module"])
    param = input["model_param"]

    model = module(param)

    return model


if __name__ == "__main__":
    input_file = command(sys.argv)
    main(input_file)
