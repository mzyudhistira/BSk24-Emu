# Introduction
This repository is a template ML pipeline that can be used to do any deep learning calculation. It is made using Keras Boston Housing as an example.
This is still a working progress. Things to do:
- [] Make analysis example
- [] Make interactive cli program

# Installation
Ensure the correct dependencies are installed. This repo is build with conda and linux in mind.
```bash
conda env create -f environment.yml
```

# Usage
1. Make sure the program is run in the correct environment
    ```bash
    conda activate ann
    ```

2. Prepare the run configuration file
    
    This can either be a TOML file (for a single run), or a TXT file (for parallel or sequential runs) containing paths to TOML files, with one path listed per line.

3. There are several files that can be run:
    - run.py

        This consists of full automatic run which can be executed with the following command.
        ```bash
        python run.py path/to/run/file.toml
        ```
        `-pl` or `-parallel` flag can be added to run the program in parallel

    - cli.py: to run the program interactively

    - analysis.ipynb: consists of typical analysis that can be done

# File Guideline
## Data
Store any data that are needed to be loaded in the project in the `data/` directory. Running the `run.py` automatically creates this directories.

## Run Config
The run configuration file is written in [TOML format](https://toml.io/en/), this is normally stored in `data/run/` directory. The run config should consists of the following data.
1. run : This include the name and note of the run
2. input: This include module, generator, and param
3. modele: This include module, genrator, and param
4. training: This include module, method, and param

An example of the run configuration file can be seen in `data/run/test.toml`

## Input
The input configuration should only include the data mentioned above. A custom generator function can be created in the input directory, and the file containing the generator function is referred to as a module. The custom generator function should only take param as the parameter and return a dictionary containing training, validation, and test data. Each returned dataset is a tuple of features and the target.

## Model
The model configuration should only include the data mentioned above. A custom generator function can be created in the model directory, and the file containing the generator function is referred to as a module. The custom generator function should only take param as the parameter and return keras model object.

## Training
The training configuration should only include the data mentioned above. A custom training method can be created in the training directory, and the file containing the training method is referred to as a module. The custom training method should only take the following arguments: 
- data: The created data from input step
- model: Model object created from model step
- training param: Parameters defined in the TOML file
- file path: File path to save the training results, initiated by the program by default

Training method is a procedure, therefore nothing is returned as everything is saved as files already.
