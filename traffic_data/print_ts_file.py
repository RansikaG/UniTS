from sktime.datasets import load_from_tsfile_to_dataframe

# Path to your .ts file
ts_file = 'cstnet.ts'
ts_file = 'cstnet/newdata_TEST.ts'
# ts_file = '/home/ransika/UniTS/dataset/MotionSenseHAR/MotionSenseHAR_TEST.ts'
# Load the .ts file into a pandas DataFrame
df, labels = load_from_tsfile_to_dataframe(ts_file, return_separate_X_and_y=True,
                                           replace_missing_vals_with='NaN')
# Print the first few rows of the time series data and the class labels (if present)
print("Time Series Data (First 5 rows):")


