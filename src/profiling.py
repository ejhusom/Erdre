import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from config import DATA_PATH


# Set path to parent directory where the .csv files are located.
#datadir = "C:/big/datasets/cnc_mill/"

#### Read data ####

# Define an empty data frame.
df = pd.DataFrame()

# Read and append the first 9 .csv files.
for i in range(1,10):
    tmp = pd.read_csv(DATA_PATH / "experiment_0" + str(i) + ".csv")
    df = df.append(tmp)

#### End of read data section ####

# Generate report.
profile = ProfileReport(df, title="Profiling Analysis", config_file="myconfig.yaml", lazy=False)

# Save report to html.
profile.to_file("profile_report.html")

