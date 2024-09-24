import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf



# Define the path to the logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.abspath(__file__))), '..', 'logs')

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
    




# Functions

def load_data(file_path):
    logger.info("Loading data from file...")
    try:
        df = pd.read_csv(file_path) #, parse_dates=['Date'])
       # df.set_index('Date', inplace=True)
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

def plot_customers_per_month(df):
    logger.info("Plot number of customer per month.")
    try:
       # df['month'] = df['Date'].dt.month
        monthly_sales = df['Sales'].resample('M').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_sales.index, monthly_sales)
        plt.title('Monthly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()

    except Exception as e:
        logger.error(f"Error plotting number of customer per month.: {e}")
        return None     

def plot_sales_per_month(df):
    logger.info("Plot number of Sales per month.")
    try:

        df['month'] = df['Date'].dt.month

        Customers_per_month = df[df['Open'] == 1].groupby('month')[['Sales']].mean().reset_index()
        plt.figure(figsize=(10, 6))
        Customers_per_month.plot(kind='bar')
        plt.title('Sales per Month')
        plt.xlabel('Months)')
        plt.ylabel('Avarage of Sales')
        plt.show()    
    except Exception as e:
        logger.error(f"Error plotting Average of Sales per month.: {e}")
        return None     




def plot_cumulative_sales(df):
    logger.info("Plotting cumulative sales over time...")
    try:
        df['CumulativeSales'] = df['Sales'].cumsum()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['CumulativeSales'])
        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting cumulative sales: {e}")

        
def plot_sales_growth_rate(df):
    logger.info("Plotting daily sales growth rate...")
    try:
        df['SalesGrowthRate'] = df['Sales'].pct_change()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['SalesGrowthRate'])
        plt.title('Daily Sales Growth Rate')
        plt.xlabel('Date')
        plt.ylabel('Growth Rate')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales growth rate: {e}")
        


def plot_day_of_week_sales(df):
    logger.info("Plotting average sales by day of week for each storetypes...")
    try:
        # df['DayOfWeek'] = df.index.dayofweek
       sns.catplot(data = df, x = 'month', y = "Sales", 
               col = 'DayOfWeek', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'StoreType', # per store type in rows
               color = 'c',
               kind= 'point') 

    except Exception as e:
        logger.error(f"Error in plotting day of week sales for each store types: {e}")



def add_holiday_column(df):
    logger.info("Adding holiday column...")
    try:
        us_holidays = holidays.US(years=[2014])
        df['is_holiday'] = df.index.to_series().apply(lambda date: date in us_holidays).astype(int)
        return df
    except Exception as e:
        logger.error(f"Error in adding holiday column: {e}")
        return df


def plot_holiday_sales_distribution(df):
    logger.info("Plotting sales distribution: Holiday vs Non-Holiday...")
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_holiday', y='Sales', data=df)
        plt.title('Sales Distribution: Holiday vs Non-Holiday')
        plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting holiday sales distribution: {e}")

def sales_trends_over_time_by_assortment_type(df):

    logger.info("Plotting Sales Trends Over Time by Assortment Type...")
    try:
        average_sales_by_assortment = df.groupby([df.index, 'Assortment'])['Sales'].mean().reset_index()

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=average_sales_by_assortment, x='Date', y='Sales', hue='Assortment')
        plt.title('Sales Trends Over Time by Assortment Type')
        plt.ylabel('Average Sales')
        plt.xlabel('Date')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting Sales Trends Over Time by Assortment Type: {e}")



def plot_holiday_effect(df):
    logger.info("Plotting holiday effect...")
    try:
        sns.catplot(data = df, x = 'DayOfWeek', y = "Customers", 
               col = 'is_holiday', # per store type in cols
               palette = 'Blues',
               hue = 'StoreType',
               row = 'StoreType', # per promo in the store in rows
               color = 'c',
               kind='point') 
    except Exception as e:
        logger.error(f"Error in plotting holiday effect: {e}")



def plot_promo_effect(df):
    logger.info("Plotting promo effect over time...")
    try:
        df.set_index('Date', inplace=True)
        monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Customers'].mean().unstack()
        monthly_promo_sales.columns = ['No Promo', 'Promo']

        monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
        plt.title('Monthly Average Sales: Promo vs No Promo')
        plt.xlabel('Date')
        plt.ylabel('Average Customer')
        plt.legend(['No Promo', 'Promo'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting promo effect: {e}")

def plot_sales_vs_promo(df):
    logger.info("Plotting sales vs promotions...")
    try:
       sns.catplot(data = df, x = 'month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = 'c',
               kind='point') 
    except Exception as e:
        logger.error(f"Error in plotting sales vs promotions: {e}")      



def plot_store_type_performance(df):
    logger.info("Plotting store type performance over time...")
    try:
        store_type_sales = df.groupby([df.index.to_period('M'), 'Store_Type'])['Sales'].mean().unstack()
        plt.figure(figsize=(15, 7))
        store_type_sales.plot()
        plt.title('Monthly Average Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(title='Store Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting store type performance: {e}")
        




def Sales_dist_by_assortment_type(df):
    logger.info("Plotting  Sales distribution by assortment type.")
    try:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='Assortment', y='Sales', palette='Set2')
        plt.title('Sales Distribution by Assortment Type')
        plt.ylabel('Sales')
        plt.xlabel('Assortment Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error in Plotting  Sales distribution by assortment type: {e}")
        
def custom_behaviour_during_open_and_close(df):
    logger.info("Plotting customer flow in open days...")
    try:
        sns.catplot(data = df, x = 'DayOfWeek', y = "Customers", 
                    col = 'Open', # per store type in cols
                    palette = 'plasma',
                    hue = 'Open',
                    # row = 'Open', # per store type in rows
                    color = 'c',
                    kind= 'point')
    except Exception as e:
        logger.error(f"Error in Plotting customer flow in open days: {e}")                
    
