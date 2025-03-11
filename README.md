# Introduction
This project is the code that I use to work on my master thesis

# Setup
The project was set up with Anaconda. To recreate the same environment, run
conda env create -f environment.yml

then enter the environment by
conda activate tf

# Usage (planned)
The pipeline of the code can be seen in the thesis.
To run the whole pipeline, use
python run.py <input_file.json>

please refer to the docummentation of each module to create the input file
in case if there is an error in one step, each module can also be run separately

## generic input data
run
name: put the name of the experiment, be concise
note: detailed desc of the experiment

input:
bsk24: default for all data, and exp for just the experimental mass (729)
bsk24_variant:
- full for full data (33000)
- sample for 3000 sample
- ext for full extrapolated data (11000)
- ext sample for 3000 sample of extrapolated data

input_module: give the module used to define the input data
input_vector: vector of the input data

model:
module: module name
name: name of the function
param: parameter of the model
