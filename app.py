# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scripts.billing_prediction import *
from scripts.llm_integration import *
from scripts.anomaly_detection import load_model_stored
# Load the language model
generator = load_model()

# Set page configuration
st.set_page_config(page_title='Energy Consumption Dashboard', layout='wide')

# Load data
daily_usage = pd.read_csv('data/processed/daily_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')
monthly_usage = pd.read_csv('data/processed/monthly_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df_usage = pd.read_csv('data/processed/daily_usage.csv', parse_dates=['Timestamp'], index_col='Timestamp')


try:
    anomaly_model = load_model_stored('models/anomaly_detection_model.pkl')
    if anomaly_model is None:
        st.warning("Anomaly detection model could not be loaded. Some features will be unavailable.")
        has_anomaly_model = False
    else:
        has_anomaly_model = True
except Exception as e:
    st.error(f"Error loading anomaly detection model: {e}")
    has_anomaly_model = False
# Sidebar for user input
st.sidebar.title("User Input")
user_budget = st.sidebar.number_input("Enter your monthly energy budget ($):", min_value=0.0, value=100.0)
user_data = st.sidebar.text_area("Describe your lifestyle and appliance usage habits:", 
                                     "I work from home and use air conditioning during the day.")

# Main dashboard
st.title("Energy Consumption Dashboard")
st.markdown("Monitor your energy usage, predict bills, and get personalized energy-saving tips.")

# Predicted Bill Display
st.header("Predicted Bill")
rate_plans = load_rate_plans('configs/rate_plans.json')
bill_predictions = predict_bill(rate_plans, monthly_usage, df_usage)
predicted_bill = bill_predictions['Flat Rate Plan']  # Assuming Flat Rate Plan for simplicity

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Bill vs Budget")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_bill,
        delta={'reference': user_budget, 'position': "top", 'relative': False},
        gauge={
            'axis': {'range': [0, max(predicted_bill, user_budget) * 1.5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, user_budget], 'color': 'lightgreen'},
                {'range': [user_budget, max(predicted_bill, user_budget) * 1.5], 'color': 'pink'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': user_budget
            }
        }
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)

with col2:
    st.subheader("Plan Comparison")
    plan_costs = pd.DataFrame.from_dict(bill_predictions, orient='index', columns=['Cost'])
    plan_fig = px.bar(plan_costs, x=plan_costs.index, y='Cost', color='Cost', 
                      color_continuous_scale='Viridis', title="Cost Under Different Plans")
    plan_fig.update_layout(xaxis_title="Energy Plans", yaxis_title="Cost ($)", coloraxis_showscale=False)
    st.plotly_chart(plan_fig, use_container_width=True)

# Energy Consumption Over Time
st.header("Energy Consumption Over Time")
fig = px.line(daily_usage, x=daily_usage.index, y='Daily Usage (kWh)', title='Daily Energy Consumption',
              labels={'x': 'Date', 'Daily Usage (kWh)': 'Energy Consumption (kWh)'})
fig.update_traces(line_color='green', line_width=2)
st.plotly_chart(fig, use_container_width=True)

# Heatmap of Energy Usage
st.header("Energy Usage Heatmap")
df_usage['Hour'] = df_usage.index.hour
df_usage['Day'] = df_usage.index.date
pivot_table = df_usage.pivot_table(values='Daily Usage (kWh)', index='Hour', columns='Day', aggfunc='sum')
heatmap_fig = px.imshow(pivot_table, aspect='auto', title='Energy Usage Heatmap',
                        labels={'x': 'Date', 'y': 'Hour of Day', 'color': 'Usage (kWh)'})
st.plotly_chart(heatmap_fig, use_container_width=True)

# Anomaly Detection
st.header("Anomaly Alerts")
# Prepare data for anomaly detection
df_features = df_usage.copy()
print("Prediction features:", df_features.columns.tolist())

df_features['Hour'] = df_features.index.hour
df_features['DayOfWeek'] = df_features.index.dayofweek
df_features['Daily Usage (kWh)'] = df_usage['Daily Usage (kWh)'] 
df_features['IsWeekend'] = df_features['DayOfWeek'] >= 5
df_features['Anomaly'] = anomaly_model.predict(df_features[['Daily Usage (kWh)', 'Hour', 'DayOfWeek', 'IsWeekend']])
anomalies = df_features[df_features['Anomaly'] == -1]

if not anomalies.empty:
    st.warning("Anomalies detected in your energy usage!")
    anomaly_times = anomalies.index.strftime('%Y-%m-%d %H:%M')
    st.write(f"Anomalies detected at: {', '.join(anomaly_times)}")
    
    # Provide explanations using LLM
    for index, row in anomalies.iterrows():
        explanation = get_anomaly_explanation(generator, index.strftime('%Y-%m-%d %H:%M'), user_data)
        st.write(f"**Anomaly at {index.strftime('%Y-%m-%d %H:%M')}:** {explanation}")
else:
    st.success("No anomalies detected.")

# Energy Saving Tips
st.header("Personalized Energy-Saving Tips")
consumption_summary = daily_usage.describe().to_string()
energy_saving_tips = get_energy_saving_tips(generator, consumption_summary, user_data)
st.write(energy_saving_tips)

# User Insights
st.header("User Insights")

# Pie Chart of Energy Consumption Distribution
st.subheader("Energy Consumption Distribution")
consumption_bins = pd.cut(daily_usage['Daily Usage (kWh)'], bins=5)
consumption_distribution = consumption_bins.value_counts().sort_index()
pie_fig = px.pie(values=consumption_distribution.values, names=[str(interval) for interval in consumption_distribution.index],
                 title='Distribution of Daily Energy Consumption')
st.plotly_chart(pie_fig, use_container_width=True)

# Additional visualizations can be added here

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit by Team 13")

