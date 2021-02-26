import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import yaml

from config import DATA_PATH_RAW, PROFILING_PATH


# Set path to parent directory where the .csv files are located.
#datadir = "C:/big/datasets/cnc_mill/"

#### Read data ####

# Define an empty data frame.
df = pd.DataFrame()

# Load parameters for finding raw files
params = yaml.safe_load(open("params.yaml"))["featurize"]

raw_subfolder = params["raw_subfolder"]
"""Subfolder of 'assets/data/raw', in where to look for data."""

dir_path = str(DATA_PATH_RAW)

if raw_subfolder != None:
    dir_path += "/" + raw_subfolder

# Read and append the first 9 .csv files.
for i in range(1,10):
    tmp = pd.read_csv(dir_path + "/" + "experiment_0" + str(i) + ".csv")
    df = df.append(tmp)

#### End of read data section ####

# Generate report.
profile = ProfileReport(df, title="Profiling Analysis", config_file="src/myconfig.yaml", lazy=False)

# Create folder for profiling report
PROFILING_PATH.mkdir(parents=True, exist_ok=True)

# Save report to html.
profile.to_file(PROFILING_PATH / "profile_report.html")

