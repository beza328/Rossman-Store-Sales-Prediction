import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.stattools import acf, pacf
import holidays
import logging


# Define the path to the logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Optional: Add a console handler if you want to see logs in the console
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # Or ERROR if you want only error messages in the console
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# Functions

def load_data(file_path):
    logger.info("Loading data from file...")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        logger.info(f"Data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def missing_values_table(df):
    logger.info("Missing values Tabel")
    try:

        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # dtype of missing values
        mis_val_dtype = df.dtypes

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    except Exception as e:
        logger.error(f"Error Missing values: {e}")
        return None
   
def merge(df1,df2):
    logger.info("Merging two datas")
    try:
    

        merged_df = df2.merge(df1, how='left', on='Store')
        return merged_df
    except Exception as e:
        logger.error(f"Error Merging two datas: {e}")
        return None    





def save_cleaned_data_to_csv(dataframe, directory, filename):
    logger.info("Save cleaned data")
    try:
        """
            Saves the cleaned DataFrame to a CSV file in the specified directory.
            If the file already exists, it will be replaced.

            Parameters:
            dataframe (pd.DataFrame): The cleaned data to save.
            directory (str): The directory where the CSV file will be saved.
            filename (str): The name of the CSV file (should include .csv extension).
            """

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the full path for the CSV file
        file_path = os.path.join(directory, filename)

        # Save the DataFrame to CSV, replacing the existing file if it exists
        dataframe.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error Saving Cleaned Datas: {e}")
        return None  
