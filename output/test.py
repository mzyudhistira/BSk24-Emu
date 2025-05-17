import importlib

import numpy as np
import pandas as pd


class Test:
    """
    Initialize Test class in the ML pipeline.

    Attributes:
        rms_dev (float): Root mean square deviation of the prediction wrt target data
        mae (float): Mean average deviation (target - prediction)
        std_diff (float): Standard difference of the difference
        output (str): Output file which contains the features, target, and difference
    """

    def __init__(self, input_obj, model_obj, train_obj, run_param) -> None:
        test_feature, test_target = input_obj.data["test"]
        prediction = model_obj.model.predict(test_feature).flatten()
        diff = test_target - prediction

        self.rms_dev = np.sqrt((diff**2).mean())
        self.mae = diff.mean()
        self.std_diff = diff.std()
        self.output = run_param["run"]["dir"] / "result.csv"

        param_header = [f"param_{i+1}" for i in range(test_feature[1].shape[0])]
        output_result = pd.DataFrame(test_feature, columns=param_header)
        output_result["target"] = test_target
        output_result["prediction"] = prediction
        output_result["difference"] = diff
        output_result.to_csv(self.output, index=False)
