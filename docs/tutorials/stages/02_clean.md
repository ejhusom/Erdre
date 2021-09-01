# Stage 2: clean

Go back to: [Documentation - Home](https://github.com/SINTEF-9012/Erdre/blob/master/docs/index.md)

Go back to: [Overview of pipeline](https://github.com/SINTEF-9012/Erdre/blob/master/docs/tutorials/03_pipeline.md)

Previous stage: [profile](https://github.com/SINTEF-9012/Erdre/blob/master/docs/tutorials/stages/01_profile.md)

This stage cleans the data for unwanted variables and samples, in addition to
encoding the target variable if the case of a classification task.


## Parameters

- `clean.target`: Name of target variable. Must correspond to the name of the
  column that contains this variable in your .csv-files.
- `clean.classification`: `True` or `False`. Specify whether we are dealing
  with a classification task or not.
- `clean.onehot_encode_target`: `True` or `False`.
- `clean.combine_files`: `True` or `False`.
- `clean.percentage_zeros_threshold`:
- `clean.correlation_metric`:
- `clean.input_max_correlation_threshold`:

## Processing

During this stage, the following operations will be performed:

- Some variables/columns might be removed from the data set, based on the
  following criteria:
    - If a variable is constant, it will be removed because it does not
      contribute any valuable information.
    - If a variables
- If we are dealing with a classification task (in which case the parameter
  `clean.classification` must be set to `True` in `params.yaml`, the target
  variable will be encoded to a number.

Next stage: [featurize](https://github.com/SINTEF-9012/Erdre/blob/master/docs/tutorials/stages/03_featurize.md)
