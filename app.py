# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio

from scripts.billing_prediction import load_rate_plans, predict_bill
from scripts.llm_integration import (
    get_anomaly_explanation,
    get_energy_saving_tips,
)
from scripts.anomaly_detection import load_model_stored, prepare_data

# Set page configuration
st.set_page_config(page_title='Energy Consumption Dashboard', layout='wide')

# Sidebar for user input
st.sidebar.title("User Input")
user_budget = st.sidebar.number_input(
    "Enter your monthly energy budget ($):", min_value=0.0, value=100.0
)


user_data = st.sidebar.text_area(
    "Describe your lifestyle and appliance usage habits:",
    "I work from home and use air conditioning during the day."
)
baseline_allowance = st.sidebar.number_input(
    "Enter your baseline allowance (kWh):", min_value=0.0, value=400.0
)

# Dictionaries to store data
daily_usage_dict = {}
monthly_usage_dict = {}
df_usage_dict = {}
anomaly_models = {}
spikes_dict = {}  # New dictionary to store spikes data

# Select Household(s)
st.sidebar.title("Household Selection")
households = ['1', '2']  # Add more households as needed
selected_households = st.sidebar.multiselect(
    "Select Households to View:", households, default=households
)

# Time range selection
st.sidebar.title("Time Range Selection")
default_start_date = pd.to_datetime('2024-08-01')  # Update with your data's start date
default_end_date = pd.to_datetime('2024-08-31')    # Update with your data's end date

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Validate date range
if start_date > end_date:
    st.sidebar.error("Error: End date must fall after start date.")
    st.stop()

# Load data for selected households
for household_id in selected_households:
    
    
    # Load daily usage data
    daily_usage = pd.read_csv(
        f'data/processed/daily_usage_{household_id}.csv',
        parse_dates=['Timestamp'],
        index_col='Timestamp'
    )
    # Display available data range
    data_start_date = daily_usage.index.min().date()
    data_end_date = daily_usage.index.max().date()
    st.sidebar.info(f"Household {household_id} - Data available from {data_start_date} to {data_end_date}")

    # Filter data by date range
    daily_usage = daily_usage.loc[start_date:end_date]

    if daily_usage.empty:
        st.warning(f"No data available for Household {household_id} in the selected date range.")
        continue  # Skip to the next household

    # Resample to monthly usage
    monthly_usage = daily_usage.resample('M').sum()
    monthly_usage = monthly_usage.reset_index()
    monthly_usage.columns = ['Timestamp', 'Monthly Usage (kWh)']  # Rename columns

    df_usage = daily_usage.copy()

    # Store in dictionaries
    daily_usage_dict[household_id] = daily_usage
    monthly_usage_dict[household_id] = monthly_usage
    df_usage_dict[household_id] = df_usage

    # Load anomaly detection model
    model_path = f'models/anomaly_detection_model_{household_id}.pkl'
    try:
        anomaly_model = load_model_stored(model_path)
        if anomaly_model is None:
            st.warning(f"Anomaly detection model for Household {household_id} could not be loaded.")
            anomaly_models[household_id] = None
        else:
            anomaly_models[household_id] = anomaly_model
    except Exception as e:
        st.error(f"Error loading anomaly detection model for Household {household_id}: {e}")
        anomaly_models[household_id] = None

    # Load spikes data
    try:
        spikes = pd.read_csv(
            f'data/processed/spikes_{household_id}.csv',
            parse_dates=['Timestamp']
        )
        # Filter spikes by date range
        spikes = spikes[(spikes['Timestamp'] >= pd.to_datetime(start_date)) & (spikes['Timestamp'] <= pd.to_datetime(end_date))]
        spikes_dict[household_id] = spikes
    except Exception as e:
        st.warning(f"No spike data available for Household {household_id}: {e}")
        spikes_dict[household_id] = pd.DataFrame()  # Empty DataFrame

# Main dashboard
st.title("Energy Consumption Dashboard")
st.markdown("Monitor your energy usage, predict bills, and get personalized energy-saving tips.")

# Predicted Bill Display
st.header("Predicted Bill")

# Aggregate predictions for all selected households
rate_plans = load_rate_plans('configs/rate_plans.json')
plan_costs_aggregate = {}

for household_id in selected_households:
    if household_id not in monthly_usage_dict:
        continue  # Skip households with no data

    monthly_usage = monthly_usage_dict[household_id]
    df_usage = df_usage_dict[household_id]

    bill_predictions = predict_bill(rate_plans, monthly_usage, df_usage, baseline_allowance)
    # Aggregate plan costs
    for plan, cost in bill_predictions.items():
        if plan in plan_costs_aggregate:
            plan_costs_aggregate[plan] += cost
        else:
            plan_costs_aggregate[plan] = cost

# Check if there are any plan costs calculated
if plan_costs_aggregate:
    # Identify the best plan
    best_plan = min(plan_costs_aggregate, key=plan_costs_aggregate.get)
    best_plan_cost = plan_costs_aggregate[best_plan]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recommended Plan")
        st.markdown(f"### **{best_plan}** is the most cost-effective plan for your consumption.")
        st.markdown(f"Estimated cost from {start_date} to {end_date}: **${best_plan_cost:.2f}**")

    with col2:
        st.subheader("Plan Cost Comparison")
        plan_costs_df = pd.DataFrame.from_dict(plan_costs_aggregate, orient='index', columns=['Cost'])
        plan_costs_df = plan_costs_df.reset_index().rename(columns={'index': 'Plan'})
        plan_fig = px.bar(
            plan_costs_df,
            x='Plan',
            y='Cost',
            color='Cost',
            color_continuous_scale='Viridis',
            title=f"Aggregate Cost Under Different Plans ({start_date} to {end_date})"
        )
        plan_fig.update_layout(xaxis_title="Energy Plans", yaxis_title="Cost ($)", coloraxis_showscale=False)
        st.plotly_chart(plan_fig, use_container_width=True)

    # Update the gauge to reflect the best plan cost
    st.subheader("Predicted Bill vs Budget")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=best_plan_cost,
        delta={'reference': user_budget, 'position': "top", 'relative': False},
        gauge={
            'axis': {'range': [0, max(best_plan_cost, user_budget) * 1.5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, user_budget], 'color': 'lightgreen'},
                {'range': [user_budget, max(best_plan_cost, user_budget) * 1.5], 'color': 'pink'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': user_budget
            }
        }
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)
else:
    st.warning("No billing data available for the selected households and date range.")

# Energy Consumption Over Time
st.header("Energy Consumption Over Time")

if daily_usage_dict:
    fig = px.line(title='Daily Energy Consumption')
    for household_id, daily_usage in daily_usage_dict.items():
        fig.add_scatter(
            x=daily_usage.index,
            y=daily_usage['Daily Usage (kWh)'],
            mode='lines',
            name=f'Household {household_id}'
        )
    fig.update_layout(xaxis_title='Date', yaxis_title='Energy Consumption (kWh)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No energy consumption data available for the selected households and date range.")

# Heatmap of Energy Usage for Each Household
st.header("Energy Usage Heatmap")
for household_id, df_usage in df_usage_dict.items():
    st.subheader(f"Household {household_id}")
    df_usage['Hour'] = df_usage.index.hour
    df_usage['Day'] = df_usage.index.date
    pivot_table = df_usage.pivot_table(
        values='Daily Usage (kWh)',
        index='Hour',
        columns='Day',
        aggfunc='sum'
    )
    heatmap_fig = px.imshow(
        pivot_table,
        aspect='auto',
        title=f'Energy Usage Heatmap for Household {household_id}',
        labels={'x': 'Date', 'y': 'Hour of Day', 'color': 'Usage (kWh)'}
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Anomaly Detection
st.header("Anomaly Alerts")

# Anomaly Detection
st.header("Anomaly Alerts")

async def display_anomalies():
    for household_id in selected_households:
        anomaly_model = anomaly_models.get(household_id)
        if anomaly_model is not None and household_id in df_usage_dict:
            df_usage = df_usage_dict[household_id]
            df_features = prepare_data(df_usage)
            df_features['Anomaly'] = anomaly_model.predict(df_features[['Daily Usage (kWh)', 'Hour', 'DayOfWeek', 'IsWeekend']])
            anomalies = df_features[df_features['Anomaly'] == -1]

            if not anomalies.empty:
                st.warning(f"Anomalies detected for Household {household_id}!")

                # Provide explanations using LLM in simple cards
                for index, row in anomalies.iterrows():
                    timestamp = index.strftime('%Y-%m-%d %H:%M')
                    explanation = await asyncio.to_thread(get_anomaly_explanation, timestamp, user_data)

                    # Display as a simple block
                    with st.container():
                        st.markdown(f"**Anomaly Detected:** {timestamp}")
                        st.markdown(f"- **Household:** {household_id}")
                        st.markdown(f"- **Explanation:** {explanation}")
                        st.markdown("---")  # Horizontal line for separation
            else:
                st.success(f"No anomalies detected for Household {household_id}.")
        else:
            st.warning(f"Anomaly detection model not available for Household {household_id}.")

asyncio.run(display_anomalies())

# Energy Saving Tips
st.header("Personalized Energy-Saving Tips")

# Generate combined consumption summary
def summarize_consumption_data(daily_usage_dict):
    summaries = []
    for household_id, daily_usage in daily_usage_dict.items():
        avg_usage = daily_usage['Daily Usage (kWh)'].mean()
        max_usage = daily_usage['Daily Usage (kWh)'].max()
        min_usage = daily_usage['Daily Usage (kWh)'].min()
        peak_days = daily_usage['Daily Usage (kWh)'].nlargest(3).index.strftime('%Y-%m-%d').tolist()
        summary = (
            f"Household {household_id}:\n"
            f"Average daily energy consumption: {avg_usage:.2f} kWh"
            f"Maximum consumption: {max_usage:.2f} kWh on {peak_days[0]}"
            f"Minimum consumption: {min_usage:.2f} kWh"
            f"Peak usage days: {', '.join(peak_days)}"
        )
        summaries.append(summary)
    return "\n".join(summaries)

consumption_summary = summarize_consumption_data(daily_usage_dict)

async def display_energy_saving_tips():
    tips = await asyncio.to_thread(get_energy_saving_tips, consumption_summary, user_data)
    st.markdown("### Here are some personalized energy-saving tips:")
    tips_list = tips.strip().split('\n')
    for tip in tips_list:
        if tip.strip() == '':
            continue
        with st.container():
            st.markdown(f"- {tip}")
            st.markdown("---")  # Horizontal line for separation

asyncio.run(display_energy_saving_tips())

# User Insights
st.header("User Insights")

# Energy Usage Spikes
st.subheader("Energy Usage Spikes")
for household_id in selected_households:
    spikes = spikes_dict.get(household_id)
    if spikes is None or spikes.empty:
        st.info(f"No spikes detected for Household {household_id}.")
        continue

    st.subheader(f"Household {household_id}")

    # Visualize spikes
    fig = px.scatter(
        spikes,
        x='Timestamp',
        y='Usage Change',
        size='Usage Change',
        color='USAGE (kWh)',
        title=f"Spikes in Energy Usage for Household {household_id}",
        labels={'Usage Change': 'Spike Size', 'USAGE (kWh)': 'Usage (kWh)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display summary statistics
    total_spikes = len(spikes)
    max_spike = spikes['Usage Change'].max()
    max_spike_time = spikes.loc[spikes['Usage Change'].idxmax(), 'Timestamp']

    st.markdown(
        f"""
        **Total Spikes Detected:** {total_spikes}  
        **Largest Spike:** {max_spike:.4f} kWh at {max_spike_time} at Household {household_id}
        """
    )

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit by Team 13")
