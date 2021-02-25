# Erdre

Erroneous data repair for Industry 4.0.

This is a project using [DVC](https://dvc.org/) for setting up a flexible and
robust pipeline for machine learning experiments.

For a quick look at an example result, go to [Evalutation](#evaluation).


## Installation and setup

Developed using Python3.8. Install the following modules:

```
pip3 install numpy pandas pandas_profiling matplotlib tensorflow sklearn plotly pyyaml dvc
```

## Add data

The data files should be placed in the folder `assets/data/raw/`. The scripts
look for `.csv`-files, so all `.csv`-files placed in this folder will be
considered as a part of the data set. If one or several of the files are not in
the expected format (and have the expected columns), the scripts will most
likely return an error message.

The data folder should look something like this:

```

assets/data/raw/
├── data/
|   └── raw/
|       ├── experiment_01.csv
|       ├── experiment_02.csv
|       ├── experiment_03.csv
|       ├── ...
|       ├── experiment_17.csv
|       ├── experiment_18.csv
|       └── README.md
├── metrics/
├── plots/
└── profiling/
```


## Usage

### Running experiments

Run experiments by executing this command on the main directory:

```
dvc repro
```

To run single stages of the pipeline, run:
```
dvc repro [STAGE NAME]
```

For example:

```
dvc repro profile     # will only run the profiling stage
dvc repro featurize   # will only run the featurize stage
```


### Adjusting parameters

Adjust the parameters in `params.yaml` to customize the machine learning model.

- `featurize.features`: List of the features you want to use as input to the
  model. See [available features](#available-features) below.
- `featurize.target`: What the target variable should be.
- `split.train_split`: Fraction of data set to use for training.
- `scale.method`: Which scaling method to use.
- `sequentialize.hist_size`: How many time steps of input to use for
  prediction.
- `sequentialize.target_size`: How large sequence (number of time steps) to
  predict per input sample.
- `train.net`: What type of neural network to use.
- `train.n_epochs`: Number of training epochs.
- `train.batch_size`: Batch size per epoch.
- `train.kernel_size`: Kernel size (only applicable when convolutional layers are
  used).


## Evaluation

When the trained model is evaluated, the program produces two plots that
visualizes the predictions on the test set. These plots are placed on the
folder `assets/plots/`, and show the following data:

- `prediction.html`: The true target values compared to the predicted target
  values. Only the first value of each target sequence is used in the plot, and
  all of this values are connected and plotted as a line. The features used in
  the model are also plotted.

![Example of prediction.](img/prediction_example.png)

- `prediction_individuals.html`: Individual predicted target sequences are
  plotted against the true values. The predicted sequences are shown in
  different colors to easily distinguish between them. Only a subset of the
  predicted target sequences are shown, in order to avoid overlapping and make
  the plot easier to interpret.

![Example of individual predictions.](img/prediction_individuals_example.png)

## Available features

Raw featurcontaines:

- `X1_ActualPosition`: actual x position of part (mm)
- `X1_ActualVelocity`: actual x velocity of part (mm/s)
- `X1_ActualAcceleration`: actual x acceleration of part (mm/s/s)
- `X1_CommandPosition`: reference x position of part (mm)
- `X1_CommandVelocity`: reference x velocity of part (mm/s)
- `X1_CommandAcceleration`: reference x acceleration of part (mm/s/s)
- `X1_CurrentFeedback`: current (A)
- `X1_DCBusVoltage`: voltage (V)
- `X1_OutputCurrent`: current (A)
- `X1_OutputVoltage`: voltage (V)
- `X1_OutputPower`: power (kW)
- `Y1_ActualPosition`: actual y position of part (mm)
- `Y1_ActualVelocity`: actual y velocity of part (mm/s)
- `Y1_ActualAcceleration`: actual y acceleration of part (mm/s/s)
- `Y1_CommandPosition`: reference y position of part (mm)
- `Y1_CommandVelocity`: reference y velocity of part (mm/s)
- `Y1_CommandAcceleration`: reference y acceleration of part (mm/s/s)
- `Y1_CurrentFeedback`: current (A)
- `Y1_DCBusVoltage`: voltage (V)
- `Y1_OutputCurrent`: current (A)
- `Y1_OutputVoltage`: voltage (V)
- `Y1_OutputPower`: power (kW)
- `Z1_ActualPosition`: actual z position of part (mm)
- `Z1_ActualVelocity`: actual z velocity of part (mm/s)
- `Z1_ActualAcceleration`: actual z acceleration of part (mm/s/s)
- `Z1_CommandPosition`: reference z position of part (mm)
- `Z1_CommandVelocity`: reference z velocity of part (mm/s)
- `Z1_CommandAcceleration`: reference z acceleration of part (mm/s/s)
- `Z1_CurrentFeedback`: current (A)
- `Z1_DCBusVoltage`: voltage (V)
- `Z1_OutputCurrent`: current (A)
- `Z1_OutputVoltage`: voltage (V)
- `S1_ActualPosition`: actual position of spindle (mm)
- `S1_ActualVelocity`: actual velocity of spindle (mm/s)
- `S1_ActualAcceleration`: actual acceleration of spindle (mm/s/s)
- `S1_CommandPosition`: reference position of spindle (mm)
- `S1_CommandVelocity`: reference velocity of spindle (mm/s)
- `S1_CommandAcceleration`: reference acceleration of spindle (mm/s/s)
- `S1_CurrentFeedback`: current (A)
- `S1_DCBusVoltage`: voltage (V)
- `S1_OutputCurrent`: current (A)
- `S1_OutputVoltage`: voltage (V)
- `S1_OutputPower`: current (A)
- `S1_SystemInertia`: torque inertia (`kg*m^2`)
- `M1_CURRENT_PROGRAM_NUMBER`: number the program is listed under on the CNC
- `M1_sequence_number`: line of G-code being executed
- `M1_CURRENT_FEEDRATE`: instantaneous feed rate of spindle


