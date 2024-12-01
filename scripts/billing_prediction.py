# scripts/billing_prediction.py

import pandas as pd
import json
import os

def load_rate_plans(config_file):
    """Load rate plans from the configuration file."""
    with open(config_file, 'r') as file:
        rate_plans = json.load(file)
    return rate_plans

def calculate_flat_rate(usage_kwh, rate):
    """Calculate cost under flat rate plan."""
    return usage_kwh * rate

def calculate_tiered_rate(usage_kwh, tiers):
    """Calculate cost under tiered rate plan."""
    cost = 0
    remaining_usage = usage_kwh
    for tier in tiers:
        if remaining_usage > tier['threshold']:
            usage_in_tier = tier['threshold']
            remaining_usage -= tier['threshold']
        else:
            usage_in_tier = remaining_usage
            remaining_usage = 0
        cost += usage_in_tier * tier['rate']
        if remaining_usage == 0:
            break
    return cost

def calculate_time_of_use_rate(df_usage, time_rates):
    """Calculate cost under time-of-use rate plan."""
    cost = 0
    for index, row in df_usage.iterrows():
        hour = index.hour
        usage = row['Daily Usage (kWh)']  # Updated column name
        for rate in time_rates:
            if rate['start_hour'] <= hour < rate['end_hour']:
                cost += usage * rate['rate']
                break
    return cost


def predict_bill(rate_plans, monthly_usage, df_usage):
    """Predict the bill under different rate plans."""
    predictions = {}
    usage_kwh = monthly_usage['Monthly Usage (kWh)'].iloc[-1]
    
    # Flat Rate Plan
    flat_rate = rate_plans['flat_rate']['rate']
    predictions['Flat Rate Plan'] = calculate_flat_rate(usage_kwh, flat_rate)
    
    # Tiered Rate Plan
    tiers = rate_plans['tiered']['tiers']
    predictions['Tiered Rate Plan'] = calculate_tiered_rate(usage_kwh, tiers)
    
    # Time-of-Use Rate Plan
    time_rates = rate_plans['time_of_use']['rates']
    predictions['Time-of-Use Rate Plan'] = calculate_time_of_use_rate(df_usage, time_rates)
    
    return predictions

if __name__ == "__main__":
    # Load processed data
    monthly_usage = pd.read_csv('data/processed/monthly_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')
    df_usage = pd.read_csv('data/processed/daily_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')
    
    # Load rate plans
    rate_plans = load_rate_plans('configs/rate_plans.json')
    
    # Predict bill
    bill_predictions = predict_bill(rate_plans, monthly_usage, df_usage)
    
    # Print predictions
    for plan, cost in bill_predictions.items():
        print(f"{plan}: ${cost:.2f}")
