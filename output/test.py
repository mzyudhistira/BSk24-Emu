import importlib

import numpy as np
import pandas as pd

from . import postprocess
from . import write


class Test:
    """
    Initialize Test class in the ML pipeline. Predicting the whole mass of the given variant.

    Attributes:
        rms_dev (float): Root mean square deviation of the prediction wrt target data
        mae (float): Mean average deviation (target - prediction)
        std_diff (float): Standard difference of the difference
        output (str): Output file which contains the features, target, and difference
    """

    def __init__(self, input_object, model_object, run_param) -> None:
        self.output = run_param["run"]["dir"] / "result.csv"

        feature, target = input_object.data["test"]
        if input_object.data["normalization_param"] is not None:
            normalization_param = input_object.data["normalization_param"]

        model_object.model.compile(optimizer="adadelta")

        prediction = model_object.model.predict(feature).flatten()
        diff = target - prediction

        self.rms_dev = np.sqrt((diff**2).mean())
        self.mae = diff.mean()
        self.std_diff = diff.std()

        feature = postprocess.unnormalize(feature, normalization_param)
        write.generate_mass_table(feature, target, prediction, self.output)
