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




def analyze_distribution(Train, Test):
    logger.info("Analyze and compare promotion distributions in training and test sets.")
    try:
    
        train_promo_distribution = Train['Promo'].value_counts(normalize=True)
        test_promo_distribution = Test['Promo'].value_counts(normalize=True)

        print("Training Set Promo Distribution:")
        print(train_promo_distribution)

        print("\nTest Set Promo Distribution:")
        print(test_promo_distribution)
        return train_promo_distribution, test_promo_distribution
    except Exception as e:
        logger.error(f"Error compararion of promotion distributions in training and test sets: {e}")
        return None    


def plot_distribution(Train, Test):

    logger.info("Plot the promotion distributions for comparison.")
    try:
        train_promo_distribution = Train['Promo'].value_counts(normalize=True)
        test_promo_distribution = Test['Promo'].value_counts(normalize=True)
        
        promo_distribution = pd.DataFrame({
            'Train': train_promo_distribution,
            'Test': test_promo_distribution
        }).fillna(0)  # Fill NaN values with 0


        promo_distribution.plot(kind='bar', figsize=(10, 6))
        plt.title('Promotion Distribution Comparison')
        plt.xlabel('Promo Status')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.legend(title='Dataset')
        plt.show()    
    except Exception as e:
        logger.error(f"Error Plot the promotion distributions for comparison.: {e}")
        return None 



