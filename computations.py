import pandas as pd
import numpy as np

# Load the historical data
df = pd.read_csv('df.csv')  # Replace with the correct path if needed

# Compute basic statistics for `time2`
time2_stats = {
    "min_time2": df['time2'].min(),
    "max_time2": df['time2'].max(),
    "mean_time2": df['time2'].mean(),
    "variance_time2": np.var(df['time2']),
    "std_dev_time2": np.std(df['time2'])
}

print(time2_stats)