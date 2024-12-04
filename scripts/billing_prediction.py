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
        threshold = tier['threshold']
        rate = tier['rate']
        if threshold == "infinite":
            usage_in_tier = remaining_usage
            cost += usage_in_tier * rate
            break
        else:
            threshold = float(threshold)
            if remaining_usage > threshold:
                usage_in_tier = threshold
                remaining_usage -= threshold
            else:
                usage_in_tier = remaining_usage
                remaining_usage = 0
            cost += usage_in_tier * rate
            if remaining_usage == 0:
                break
    return cost

def calculate_time_of_use_rate(df_usage, time_rates):
    """Calculate cost under time-of-use rate plan."""
    cost = 0
    for index, row in df_usage.iterrows():
        hour = index.hour
        usage = row['Daily Usage (kWh)']  # Update the column name as per your data
        for rate in time_rates:
            start_hour = rate['start_hour']
            end_hour = rate['end_hour']
            rate_value = rate['rate']
            if start_hour < end_hour:
                if start_hour <= hour < end_hour:
                    cost += usage * rate_value
                    break
            else:  # For time periods that cross midnight
                if hour >= start_hour or hour < end_hour:
                    cost += usage * rate_value
                    break
    return cost

def predict_bill(rate_plans, monthly_usage, df_usage, baseline_allowance=400):
    """Predict the bill under different rate plans."""
    predictions = {}
    if monthly_usage.empty:
        print("Monthly usage data is empty. Cannot predict bill.")
        predictions['Flat Rate Plan'] = 0.0
        predictions['Tiered Rate Plan'] = 0.0
        predictions['Time-of-Use Rate Plan'] = 0.0
        return predictions

    usage_kwh = monthly_usage['Monthly Usage (kWh)'].iloc[-1]
    total_usage = usage_kwh
    # Flat Rate Plan
    flat_rate = rate_plans['flat_rate']['rate']
    flat_cost = calculate_flat_rate(usage_kwh, flat_rate)
    predictions['Flat Rate Plan'] = flat_cost

    # Tiered Rate Plan
    tiers = rate_plans['tiered']['tiers']
    # Update tiers with actual thresholds
    for tier in tiers:
        if tier['threshold'] == "baseline_kWh":
            tier['threshold'] = baseline_allowance
    tiered_cost = calculate_tiered_rate(usage_kwh, tiers)
    predictions['Tiered Rate Plan'] = tiered_cost

    # Time-of-Use Rate Plan
    time_rates = rate_plans['time_of_use']['rates']
    # Prepare hourly usage data
    df_usage_hourly = df_usage.copy()
    df_usage_hourly['Hour'] = df_usage_hourly.index.hour
    time_of_use_cost = calculate_time_of_use_rate(df_usage_hourly, time_rates)
    predictions['Time-of-Use Rate Plan'] = time_of_use_cost

    return predictions

if __name__ == "__main__":
    # Load processed data
    monthly_usage = pd.read_csv('data/processed/monthly_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')
    df_usage = pd.read_csv('data/processed/daily_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')

    # Load rate plans
    rate_plans = load_rate_plans('configs/rate_plans.json')

    # Define baseline allowance (replace with actual value)
    baseline_allowance = 400  # Replace with actual baseline_kWh for your area and season

    # Predict bill
    bill_predictions = predict_bill(rate_plans, monthly_usage, df_usage, baseline_allowance)

    # Print predictions
    for plan, cost in bill_predictions.items():
        print(f"{plan}: ${cost:.2f}")