# Add raw data here!

The raw data should be placed in this subfolder (`Erdre/assets/data/raw`).

Example:

```
assets/
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

If you want to keep data in separate subfolders, make a subfolder in
`Erdre/assets/data/raw` and enter the subfolder name as the parameter
`featurize.dataset` in `params.yaml`.

Example with a subfolder called `experiment1`:

```
assets/
├── data/
|   └── raw/
|       ├── experiment1/
|       |   ├── experiment_01.csv
|       |   ├── experiment_02.csv
|       |   ├── experiment_03.csv
|       |   ├── ...
|       |   ├── experiment_17.csv
|       |   └── experiment_18.csv
|       └─── README.md
├── metrics/
├── plots/
└── profiling/
```

And then set the subfolder name in `params.yaml`:

```
...

featurize:
  dataset: experiment1
  features:
    - ...
    - ...

...

```

