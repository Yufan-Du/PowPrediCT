## PowPredict
This is an official repo for "PowPrediCT: Cross-Stage Power Prediction with Circuit-Transformation-Aware Learning" (DAC 2024)
### Test Result

Each test result from the cross-validation is recorded in the file "result.log", with a performance summary table for all seven circuits attached at the end of this file.

### Code

#### Train

Folder Train_phase1

Folder Train_phase2&3

#### Test

Folder Test_phase1&2&3

#### Descriptions

get_data.py: read and initialize the dataset.

config.py: define some numbers and parameters. (learning rate, scaling factor...)

model.py/model_cnn.py/model_cts.py: Pytorch models.

train_xxx.py: Python scripts for training.

test_all_phases.py: final testing script, from which we could get the above-mentioned "result.log" file.
