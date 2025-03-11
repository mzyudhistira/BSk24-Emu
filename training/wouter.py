# External Library
import numpy as np
from scipy import stats
from keras.callbacks import ModelCheckpoint, Callback

import time
import psutil
import GPUtil

# Internal Library
from model import *
from config import *

training_data_dir = DATA_DIR / "training"


class ResourceUsageCallback(Callback):
    def __init__(self):
        super(ResourceUsageCallback, self).__init__()
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []

    def on_epoch_end(self, epoch, logs=None):
        # Measure CPU usage
        cpu = psutil.cpu_percent(interval=1)
        self.cpu_usage.append(cpu)

        # Measure GPU usage
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assuming using the first GPU
            gpu_load = gpu.load * 100
        else:
            gpu_load = 0

        self.gpu_usage.append(gpu_load)

        print(f"Epoch {epoch + 1}: CPU Usage: {cpu}%, GPU Usage: {gpu_load}%")


def write_loss(history, name):
    np.savetxt(training_data_dir / "loss" / f"{name}.loss.dat", history.history["loss"])
    np.savetxt(
        training_data_dir / "loss" / f"{name}.val_loss.dat", history.history["val_loss"]
    )


def write_perf_log(
    log,
    name,
    file=training_data_dir / "performance_benchmark" / "individual_training.dat",
):
    new_line = (
        f"\n{name};{log['training_time']};{log['avg_cpu_usage']};{log['avg_gpu_usage']}"
    )

    with open(file, "a") as f:
        f.write(new_line)


def model_training(
    training_name,
    model,
    training_data,
    validation_data,
    batch,
    epoch,
    weights_file="",
    save_perf_log=False,
    save_loss=True,
):
    """
    A standardized function to train the model, several benchmarking tools are used.

    Input:
        - training_name: name of the training (will be used as saved file's name)
        - model: Keras model object
        - training_data: training data, make sure to use the last column for the target
        - validation_data: validation data, make sure to use the last column for the target
        - batch: Number of batch
        - epoch: Number of epoch
        - weights_file: load a pre-trained model with a saved weights
        - save_perf_log: write a log of training performance, only set to true if doing an individual training
        - save_loss: save the model's loss data

    Output:
        - history: Keras history object detailing the training step
        - last_weights: File which contains the weights of each neuron from the last training
        - best_weights: File which contains the best weights of each neuron from the trainings
        - performance_log: training performance
    """

    best_weights = training_data_dir / "weight_best" / f"{training_name}.weights.h5"
    last_weights = training_data_dir / "weight_training" / f"{training_name}.weights.h5"

    feature = training_data[:, :-1]
    target = training_data[:, -1]
    val_feature = validation_data[:, :-1]
    val_target = validation_data[:, -1]

    if weights_file != "":
        model.load_weights(weights_file)

    start_time = time.time()
    model_checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_best_only=True,
        monitor="val_mae",
        mode="min",
        filepath=best_weights,
    )
    resource_monitor = ResourceUsageCallback()

    history = model.fit(
        feature,
        target,
        validation_data=(val_feature, val_target),
        batch_size=batch,
        epochs=epoch,
        verbose=2,
        callbacks=[model_checkpoint_callback],
    )

    model.save_weights(last_weights)

    end_time = time.time()

    training_time = end_time - start_time
    avg_cpu_usage = np.mean(resource_monitor.cpu_usage)
    avg_gpu_usage = np.mean(resource_monitor.gpu_usage)
    performance_log = {
        "training_time": training_time,
        "avg_cpu_usage": avg_cpu_usage,
        "avg_gpu_usage": avg_gpu_usage,
    }

    # Saving the performance log and loss data
    if save_perf_log == True:
        write_perf_log(performance_log, training_name)

    if save_loss == True:
        write_loss(history, training_name)

    return history, last_weights, best_weights, performance_log


def fine_grain_training(
    model,
    train_dat,
    val_dat,
    batch_number=[32, 16, 4],
    epoch_number=[1000, 500, 500],
    training_name="",
):
    """
    Perform three training steps on our model.
      1. Rough training with prespecified optimizer in the model (rmsprop)
         and large (=32) batchsize.
      2. Less rough training with same optimizer, smaller batchsize (=16).
         This step starts from the last model obtained in the previous step.
      3. Fine-grained training with 'adagrad' optimizer for very small
         batchsize (=4). This step starts from the best model obtained in step
         2, NOT the last one.

      Input:
        model     : Keras model object
        training_name: a name to indicate the training
        train_dat : training data
        val_dat   : validation data
        batch_number: the number of batch of each training process
        epoch_number: the number of epoch of each training process


      Output:
        history1/2/3: Keras History objects of all three steps
        bestfname: the best weights throughout the training process
    """

    if training_name == "":
        training_names = [
            f"test.batch={str(batch_number[i])}.epoch={epoch_number[i]}.stage{i+1}"
            for i in range(3)
        ]

    else:
        training_names = [
            f"{training_name}.batch={str(batch_number[i])}.epoch={epoch_number[i]}.stage{i+1}"
            for i in range(3)
        ]

    performance_logs = [i for i in range(3)]
    history = [i for i in range(3)]

    history[0], lastfname, bestfname, performance_logs[0] = model_training(
        training_names[0], model, train_dat, val_dat, batch_number[0], epoch_number[0]
    )

    history[1], lastfname, bestfname, performance_logs[1] = model_training(
        training_names[1],
        model,
        train_dat,
        val_dat,
        batch_number[1],
        epoch_number[1],
        weights_file=lastfname,
    )

    model = wouter_model(len(train_dat[0][:-1]), "adagrad")

    history[2], lastfname, bestfname, performance_logs[2] = model_training(
        training_names[2],
        model,
        train_dat,
        val_dat,
        batch_number[2],
        epoch_number[2],
        weights_file=bestfname,
    )

    total_training_time = np.sum([item["training_time"] for item in performance_logs])
    avg_cpu_usage = np.mean([item["avg_cpu_usage"] for item in performance_logs])
    avg_gpu_usage = np.mean([item["avg_gpu_usage"] for item in performance_logs])
    performance_log = {
        "training_time": total_training_time,
        "avg_cpu_usage": avg_cpu_usage,
        "avg_gpu_usage": avg_gpu_usage,
    }

    write_perf_log(
        performance_log,
        training_name,
        training_data_dir / "performance_benchmark" / "fine_grain_training.dat",
    )

    return history[0], history[1], history[2], bestfname
