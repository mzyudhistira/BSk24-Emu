import importlib

import numpy as np
import pandas as pd


class Test:
    """
    Initialize Test class in the ML pipeline. Predicting the whole mass of the given variant.

    Attributes:
        rms_dev (float): Root mean square deviation of the prediction wrt target data
        mae (float): Mean average deviation (target - prediction)
        std_diff (float): Standard difference of the difference
        output (str): Output file which contains the features, target, and difference
    """

    def __init__(self, input_obj, model_obj, train_obj, run_param) -> None:
        train_data = input_obj.data["train"]
        val_data = input_obj.data["val"]
        test_data = input_obj.data["test"]
        total_feature = np.concatenate([train_data[0], val_data[0], test_data[0]])
        total_target = np.concatenate([train_data[1], val_data[1], test_data[1]])

        prediction = model_obj.model.predict(total_feature).flatten()
        diff = total_target - prediction


        self.rms_dev = np.sqrt((diff**2).mean())
        self.mae = diff.mean()
        self.std_diff = diff.std()
        self.output = run_param["run"]["dir"] / "result.csv"

        param_header = [f"param_{i+1}" for i in range(total_feature[1].shape[0])]
        output_result = pd.DataFrame(total_feature, columns=param_header)
        output_result["target"] = total_target
        output_result["prediction"] = prediction
        output_result["difference"] = diff
        output_result.to_csv(self.output, index=False)
