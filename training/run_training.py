import importlib


def main(input):
    input_data = input["input"]["data"]
    model = input["model"]["model"]

    training_module = importlib.import_module(input["training"]["training_module"])
    training_method = getattr(training_module, input["training"]["training_method"])
    training_params = input["training"]["training_params"]

    model_weights, loss, val_loss = training_method(input_data, model, training_params)

    return model_weights, loss, val_loss


if __name__ == "__main__":
    input_file = command(sys.argv)
    main(input_file)
