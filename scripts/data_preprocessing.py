import pandas as pd
import os

def load_data(file_path):
    """Load and preprocess the raw data."""
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Combine DATE and START TIME into a single Timestamp column
    df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['START TIME'])

    # Set Timestamp as the index
    df.set_index('Timestamp', inplace=True)

    # Drop unnecessary columns (e.g., DATE, START TIME, END TIME, NOTES)
    df.drop(columns=['DATE', 'START TIME', 'END TIME', 'NOTES'], inplace=True, errors='ignore')

    return df

def detect_spikes(df, threshold=0.1):
    """Detect spikes in energy usage based on a threshold."""
    df['Usage Change'] = df['USAGE (kWh)'].diff().fillna(0).abs()
    spikes = df[df['Usage Change'] > threshold].reset_index()
    # Ensure the correct number of columns
    spikes = spikes[['Timestamp', 'USAGE (kWh)', 'Usage Change']]
    return spikes

def process_daily_usage(df):
    """Aggregate data to daily usage."""
    daily_usage = df.resample('D')['USAGE (kWh)'].sum()
    daily_usage = daily_usage.reset_index()
    daily_usage.columns = ['Timestamp', 'Daily Usage (kWh)']
    return daily_usage

def process_monthly_usage(df):
    """Aggregate data to monthly usage."""
    monthly_usage = df.resample('M')['USAGE (kWh)'].sum()
    monthly_usage = monthly_usage.reset_index()
    monthly_usage.columns = ['Timestamp', 'Monthly Usage (kWh)']
    return monthly_usage

def save_processed_data(daily_usage, monthly_usage, spikes, household_id):
    """Save the processed data."""
    output_dir = 'data/processed/'
    os.makedirs(output_dir, exist_ok=True)
    daily_usage.to_csv(f'{output_dir}daily_usage_{household_id}.csv', index=False)
    monthly_usage.to_csv(f'{output_dir}monthly_usage_{household_id}.csv', index=False)
    spikes.to_csv(f'{output_dir}spikes_{household_id}.csv', index=False)

if __name__ == "__main__":
    households = ['1', '2']  # Add more households if needed

    for household_id in households:
        file_path = f'data/raw/energy_consumption_household{household_id}.csv'
        print(f"Processing data for Household {household_id}...")

        try:
            # Load and preprocess data
            df = load_data(file_path)

            # Detect spikes
            spikes = detect_spikes(df)

            # Process daily and monthly usage
            daily_usage = process_daily_usage(df)
            monthly_usage = process_monthly_usage(df)

            # Save processed data
            save_processed_data(daily_usage, monthly_usage, spikes, household_id)
            print(f"Data processed and saved for Household {household_id}.\n")

        except Exception as e:
            print(f"Error processing data for Household {household_id}: {e}")
