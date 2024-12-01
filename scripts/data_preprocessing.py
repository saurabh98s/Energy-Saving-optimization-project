# scripts/data_preprocessing.py

import pandas as pd
import os

def load_data(file_path):
    """Load the raw energy consumption data."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the raw data."""
    # Combine DATE and START TIME into a single datetime column
    df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['START TIME'])
    
    # Sort the dataframe by Timestamp
    df = df.sort_values('Timestamp')
    
    # Set Timestamp as the index
    df.set_index('Timestamp', inplace=True)
    
    # Drop unnecessary columns
    df = df.drop(columns=['TYPE', 'DATE', 'START TIME', 'END TIME', 'COST', 'NOTES'])
    
    # Handle missing values if any
    df = df.fillna(method='ffill')
    
    return df

def aggregate_data(df):
    """Aggregate data to daily and monthly usage."""
    # Resample to daily usage
    daily_usage = df.resample('D').sum()
    daily_usage.columns = ['Daily Usage (kWh)']
    
    # Resample to monthly usage
    monthly_usage = df.resample('M').sum()
    monthly_usage.columns = ['Monthly Usage (kWh)']
    
    return daily_usage, monthly_usage

def save_processed_data(daily_usage, monthly_usage):
    """Save the aggregated data to processed data directory."""
    os.makedirs('data/processed', exist_ok=True)
    daily_usage.to_csv('data/processed/daily_usage.csv')
    monthly_usage.to_csv('data/processed/monthly_usage.csv')
    print("Processed data saved successfully.")

if __name__ == "__main__":
    # Load the data
    df = load_data('data/raw/energy_consumption.csv')
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Aggregate the data
    daily_usage, monthly_usage = aggregate_data(df)
    
    # Save the processed data
    save_processed_data(daily_usage, monthly_usage)
